# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Run the §1 table on the linear subject over Tier-1 cells and write the result.

    PYTHONPATH=src python -m latent_monitor.run_table --out results/latent_monitor_tier1

Writes ``table.json`` (every statistic, threshold and attribution), ``table.md``
(the human-readable table) and ``adjustments.json`` (re-whitening, patching
and stage-refit outcomes). Every cell is scored against its pre-registered
expectation; the markdown says which rows matched and which did not, and why.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .adjust import activation_patch, damage_patch, refit_stage, rewhiten
from .designed import DesignedCell, DesignedFamily
from .linear_subject import LinearSubject
from .lookup import attribute, calibrate
from .reference import fit_reference
from .statistics import cell_statistics
from .tier1 import TARGET_NAMES, all_cells, geometry_cells, reference_cell, sigma_covariance_cells, sigma_structural_cells

EXPECTED = {"sigma_cov": "sigma_cov", "sigma_struct": "sigma_struct", "geometry": "geometry",
            "event": "event", "event_in_span": "event_in_span", "designed": None}

#: Outcomes that are documented rather than wrong. Timing jitter applied to a trace whose
#: noise has a shared cross-channel component *decorrelates* that component, so its
#: noise-only signature is a covariance change — N by contract, Σ-covariance in the latent.
DOCUMENTED = {"sigma_struct:timing_jitter": {"sigma_cov", "sigma_struct"}}


def _jsonable(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def run(out: Path, n_channels: int = 8, n_samples: int = 256, latent_dim: int = 6,
        n_fit: int = 200, n_eval: int = 60, n_noise: int = 100, designed_norm: float = 3.0, seed: int = 0) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    ref_cell = reference_cell(n_channels=n_channels, n_samples=n_samples)
    fit_ids = np.arange(0, n_fit)
    eval_ids = np.arange(1000, 1000 + n_eval)
    noise_ids = np.arange(5000, 5000 + n_noise)
    X, T = ref_cell.batch(fit_ids)
    subject = LinearSubject.fit(X, T, ref_cell.geometry, ref_cell.implied_whitener(), latent_dim=latent_dim,
                                seed=seed, target_names=TARGET_NAMES)
    ref = fit_reference(subject, ref_cell, eval_ids, noise_ids, seed=seed)
    thr = calibrate(ref)

    cells = all_cells(ref_cell) + [
        DesignedCell(ref_cell, subject, ref, DesignedFamily("output_null", designed_norm, seed)),
        DesignedCell(ref_cell, subject, ref, DesignedFamily("output_aligned", designed_norm, seed)),
    ]
    rows = []
    for cell in cells:
        s = cell_statistics(ref, subject, ref_cell, cell, eval_ids, noise_ids)
        a = attribute(s, thr)
        expected = EXPECTED[cell.moved]
        if cell.moved == "designed":
            expected = cell.label.split(":")[1]
        status = "match" if a.label == expected else ("documented" if a.label in DOCUMENTED.get(cell.label, set()) else "MISMATCH")
        rows.append({"cell": cell.label, "moved": cell.moved, "expected": expected, "attributed": a.label,
                     "status": status, "reason": a.reason, "adjustment": a.adjustment,
                     "statistics": s.summary(), "evidence": _jsonable(a.evidence)})

    # --- adjustments
    adjustments: dict[str, Any] = {"rewhiten": {}, "activation_patch": {}, "damage_patch": {}, "refit_stage": {}}
    for cell in sigma_covariance_cells(ref_cell):
        s0 = cell_statistics(ref, subject, ref_cell, cell, eval_ids, noise_ids)
        adj, info = rewhiten(subject, cell.noise_batch(np.arange(6000, 6200)))
        ref2 = fit_reference(adj, cell, eval_ids, noise_ids, seed=seed)
        s1 = cell_statistics(ref2, adj, cell, cell, eval_ids, noise_ids)
        Xc, Tc = cell.batch(eval_ids)
        before = np.mean(np.abs(subject.outputs(Xc, cell.geometry) - Tc), axis=0) / ref.consequence_ref
        after = np.mean(np.abs(adj.outputs(Xc, cell.geometry) - Tc), axis=0) / ref.consequence_ref
        adjustments["rewhiten"][cell.label] = {
            "kappa_correction": info["kappa_correction"],
            "z_var_ratio_mean_before": float(np.mean(s0.z_var_ratio)), "z_var_ratio_mean_after": float(np.mean(s1.z_var_ratio)),
            "psd_dev_before": s0.psd_dev_smooth, "psd_dev_after": s1.psd_dev_smooth,
            "consequence_ratio_before": before.tolist(), "consequence_ratio_after": after.tolist(),
        }
    Xclean, _ = ref_cell.batch(eval_ids)
    for cell in sigma_structural_cells(ref_cell):
        Xp, Tp = cell.batch(eval_ids)
        adjustments["activation_patch"][cell.label] = activation_patch(subject, Xp, Xclean, Tp, cell.geometry)
        adjustments["damage_patch"][cell.label] = damage_patch(subject, Xp, Xclean, Tp, cell.geometry)
        entry = {}
        Xr, Tr = cell.batch(np.arange(300, 400))
        base = np.mean(np.abs(subject.outputs(Xp, cell.geometry) - Tp), axis=0) / ref.consequence_ref
        for stage in ("channel", "token", "output"):
            adj = refit_stage(subject, stage, Xr, Tr, cell.geometry)
            entry[stage] = (np.mean(np.abs(adj.outputs(Xp, cell.geometry) - Tp), axis=0) / ref.consequence_ref).tolist()
        adjustments["refit_stage"][cell.label] = {"before": base.tolist(), **entry}
    for cell in geometry_cells(ref_cell):
        Xg, Tg = cell.batch(np.arange(300, 400))
        Xe, Te = cell.batch(eval_ids)
        base = np.mean(np.abs(subject.outputs(Xe, cell.geometry) - Te), axis=0) / ref.consequence_ref
        entry = {"before": base.tolist()}
        for stage in ("token", "channel", "output"):
            adj = refit_stage(subject, stage, Xg, Tg, cell.geometry)
            entry[stage] = (np.mean(np.abs(adj.outputs(Xe, cell.geometry) - Te), axis=0) / ref.consequence_ref).tolist()
        adjustments["refit_stage"][cell.label] = entry

    result = {
        "latent_monitor_version": __version__,
        "config": {"n_channels": n_channels, "n_samples": n_samples, "latent_dim": latent_dim, "n_fit": n_fit,
                   "n_eval": n_eval, "n_noise": n_noise, "designed_norm": designed_norm, "seed": seed},
        "subject": {"kind": "LinearSubject", **_jsonable(subject.meta)},
        "reference": {"fisher_rank": ref.fisher_rank, "k_out": ref.k_out, "consequence_ref": ref.consequence_ref.tolist(),
                      "null_mean_shift_quantiles": ref.null_mean_shift_quantiles, "null_dz_norm_quantiles": ref.null_dz_norm_quantiles},
        "thresholds": thr.to_dict(),
        "rows": rows,
        "adjustments": _jsonable(adjustments),
        "n_match": sum(r["status"] == "match" for r in rows),
        "n_documented": sum(r["status"] == "documented" for r in rows),
        "n_mismatch": sum(r["status"] == "MISMATCH" for r in rows),
    }
    (out / "table.json").write_text(json.dumps(result, indent=2, default=_jsonable))
    (out / "adjustments.json").write_text(json.dumps(_jsonable(adjustments), indent=2))
    (out / "table.md").write_text(render_markdown(result))
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [f"# Latent-monitoring table — linear subject, Tier-1 cells (latent_monitor {result['latent_monitor_version']})", ""]
    c = result["config"]
    lines.append(f"C = {c['n_channels']}, N = {c['n_samples']}, k = {c['latent_dim']}; {c['n_fit']} fit / {c['n_eval']} eval events, {c['n_noise']} noise-only records; seed {c['seed']}.")
    lines.append(f"Fisher rank {result['reference']['fisher_rank']}, k_out {result['reference']['k_out']}. "
                 f"{result['n_match']} match, {result['n_documented']} documented, {result['n_mismatch']} mismatch.")
    lines += ["", "| cell | moved | expected | attributed | status | mean-shift | z-var ratio | psd dev | chan-corr | out-of-span | layer peak | consequence ratio |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    for r in result["rows"]:
        s = r["statistics"]
        lines.append(f"| {r['cell']} | {r['moved']} | {r['expected']} | **{r['attributed']}** | {r['status']} | {s['mean_shift_norm']:.1f} | "
                     f"{s['z_var_ratio_mean']:.2f} | {s['psd_dev_smooth']:.2f}/{s['psd_dev_line']:.2f} | {s['residual_chan_corr_shift']:.2f} | "
                     f"{s['out_of_span_fraction']:.2f} | {s['layer_peak']} | {s['consequence_ratio']} |")
    lines += ["", "Thresholds (calibrated once on the reference null): " + ", ".join(f"{k}={v:.3g}" for k, v in result["thresholds"].items()), ""]
    lines += ["## Adjustments", "", "### Re-whitening (Σ-covariance cells)", "", "| cell | κ correction | z-var ratio before → after | consequence ratio before → after |", "|---|---:|---|---|"]
    for k, v in result["adjustments"]["rewhiten"].items():
        lines.append(f"| {k} | {v['kappa_correction']['kappa']:.2f} | {v['z_var_ratio_mean_before']:.2f} → {v['z_var_ratio_mean_after']:.2f} | "
                     f"{[round(x, 2) for x in v['consequence_ratio_before']]} → {[round(x, 2) for x in v['consequence_ratio_after']]} |")
    lines += ["", "### Stage-restricted refit (consequence ratio; 1.0 = reference)", "", "| cell | before | channel | token | output |", "|---|---|---|---|---|"]
    for k, v in result["adjustments"]["refit_stage"].items():
        f = lambda key: [round(x, 2) for x in v[key]] if key in v else "—"
        lines.append(f"| {k} | {f('before')} | {f('channel')} | {f('token')} | {f('output')} |")
    lines += ["", "### Activation patching (fraction of the consequence gap recovered by substituting the clean stage)", ""]
    for k, v in result["adjustments"]["activation_patch"].items():
        lines.append(f"- {k}: " + ", ".join(f"{s}={d['recovery']:.2f}" for s, d in v.items() if s not in ("clean",)))
    lines += ["", "For an input-side corruption every stage recovers fully: the linear subject has no stage that *creates* damage. "
              "Stage localisation (C5) is therefore a question for the nonlinear subjects; here the informative repair contrast is the stage refit above.", ""]
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("results/latent_monitor_tier1"))
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=6)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--n-noise", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    r = run(a.out, a.channels, a.samples, a.latent_dim, n_eval=a.n_eval, n_noise=a.n_noise, seed=a.seed)
    print(f"{r['n_match']} match, {r['n_documented']} documented, {r['n_mismatch']} mismatch -> {a.out}")
    for row in r["rows"]:
        print(f"  {row['status']:10s} {row['cell']:28s} -> {row['attributed']}")
    return 0 if r["n_mismatch"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
