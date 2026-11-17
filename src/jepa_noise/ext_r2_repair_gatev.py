"""Re-score frozen Gate V holdout transport under a 2 × 2 optics grid."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np

from . import cherenkov_validation as cv
from . import ext_r2 as r2
from . import lucid_bombs as lb
from . import r2_generator as gen
from . import r2_geometry as rg
from .ext_r2_repair_diagnostic_helpers import GATEV_CACHE, trace_event
from .ext_r2_repair_diagnostics import DETECTORS, EXTENSION, ROOT, write_json
from .ext_r2_repair_probes import safe_output

GRID = {(label, mode): cv.AnalyticConfig(attenuation_m=length, angular_mode=mode)
        for label, length in (("50m", 50.0), ("off", None))
        for mode in ("cos", "spherical")}
METRICS = ("yield_ratio", "module_pattern_pearson", "log_pattern_pearson",
           "pattern_pearson", "time_median_abs_ns")


def _detector_parameters(geometry, cfg):
    derived = rg.derived_values(geometry)
    rule = gen.ROUND1_TIME_RULE if geometry.detector == "wcte" else gen.time_rule_for(geometry)
    return {
        "n_active_pmts": geometry.n_pmt, "pmt_radius_m": geometry.sensor_radius_m,
        "detector_radius_m": geometry.radius_m, "geometry": geometry.shape,
        "prompt_window_ns": list(rule.prompt_window_ns), "nhits_n": rule.n,
        "time_scale_ns": rule.time_scale_ns, "dark_rate_hz": cfg.dark_rate_hz,
        "qe_curve_nm": list(cfg.qe_curve_nm), "qe_curve_values": list(cfg.qe_curve),
        "collection_efficiency": cfg.ce, "group_index": cfg.n_group,
        "tts_sigma_ns": cfg.tts_sigma_ns, "r2_geometry_derived": derived,
    }


def score_holdout(item):
    detector, event_id, name = item
    geometry = rg.load_r2_geometry(detector)
    cfg = gen.load_r2_config()
    cache = GATEV_CACHE / f"{detector}__{name}"
    event = cv.load_event_npz(cache) if cache.is_file() else trace_event(detector, name, cfg)
    if int(event.event_id) != event_id:
        raise ValueError(f"{detector}/{name}: transport identity mismatch")
    _, observed, _ = cv._observed(event, geometry, cfg)
    delta = geometry.positions_m - event.vertex_m
    distance = np.linalg.norm(delta, axis=1)
    incidence = np.degrees(np.arccos(np.clip(
        np.abs(np.einsum("ij,ij->i", delta, geometry.directions)) / distance, 0.0, 1.0)))
    coords = {"distance_m": distance, "incidence_deg": incidence,
              "height_m": geometry.positions_m[:, 2]}
    edges = {"distance_m": np.linspace(0.0, rg.longest_path_m(geometry), 21),
             "incidence_deg": np.linspace(0.0, 90.0, 19),
             "height_m": np.linspace(-geometry.half_height_m, geometry.half_height_m, 17)}
    records, binned = {}, {}
    for (label, mode), analytic in GRID.items():
        key = f"{label}_{mode}"
        records[key] = cv.compare_event(geometry, cfg, event, analytic=analytic)
        mu, _, _ = cv.analytic_prediction(geometry, cfg, event, analytic)
        binned[key] = {
            axis: {"transport": np.histogram(values, bins=edges[axis], weights=observed)[0].tolist(),
                   "analytic": np.histogram(values, bins=edges[axis], weights=mu)[0].tolist()}
            for axis, values in coords.items()
        }
    return {"detector": detector, "holdout": name, "event_id": event_id,
            "source": str(cache) if cache.is_file() else "validated straight-ray sphere tracer",
            "records": records, "binned": binned, "edges": {k: v.tolist() for k, v in edges.items()}}


def reference(detector):
    p = lb._repo_root() / f"results/round2/20261112_x0/{detector}_gatev.json"
    if p.is_file():
        x = json.loads(p.read_text())
        return x.get("gate_v", x)
    return None


def collect(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(det, i, name) for det in DETECTORS for i, name in enumerate(r2.GATE_V_HOLDOUT)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = {}
        for event in pool.map(score_holdout, jobs, chunksize=1):
            results.setdefault(event["detector"], []).append(event)
    summary = {}
    cfg = gen.load_r2_config()
    for detector, rows in results.items():
        geometry = rg.load_r2_geometry(detector)
        metrics = {}
        for label, mode in GRID:
            key = f"{label}_{mode}"
            metrics[key] = {metric: float(np.median([row["records"][key][metric] for row in rows]))
                            for metric in METRICS}
        ref = reference(detector)
        if ref is not None:
            ref = {"yield_ratio": float(np.median([x["yield_ratio"] for x in ref["rows"]])),
                   "module_pattern_pearson": float(np.median(
                       [x["module_pattern_pearson"] for x in ref["rows"]]))}
        summary[detector] = {"median": metrics, "legacy_reference": ref,
                             "detector_parameters": _detector_parameters(geometry, cfg),
                             "n_events": len(rows), "rows": rows}
    summary = {"extension": EXTENSION, "registered": False, "detectors": summary}
    write_json(out / "gatev_reconciliation.json", summary)
    text = ["# Gate V optics reconciliation", "",
            "| Detector | Attenuation | Angular acceptance | LUCiD / analytic yield | Module r | Log-charge r |",
            "|---|---|---|---:|---:|---:|"]
    for detector, result in summary["detectors"].items():
        for key, value in result["median"].items():
            label, mode = key.split("_")
            text.append(f"| {detector.upper()} | {label} | {mode} | {value['yield_ratio']:.4f} "
                        f"| {value['module_pattern_pearson']:.4f} | {value['log_pattern_pearson']:.4f} |")
    text.extend(["", "Each row uses the same ten holdouts and active sensor mapping. "
                 "HK transport is replayed with a straight-ray spherical-PMT tracer. "
                 "The original Gate V values are retained in the JSON for comparison.", ""])
    (out / "GATEV_RECONCILIATION.md").write_text("\n".join(text))


def plots(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
    axes_names = ("distance_m", "incidence_deg", "height_m")
    axes_labels = ("PMT distance (m)", "incidence angle (deg)", "PMT z (m)")
    for i, detector in enumerate(("sk", "hk")):
        rows = result["detectors"][detector]["rows"]
        for ax, name, xlabel in zip(axes[i], axes_names, axes_labels):
            edges = np.asarray(rows[0]["edges"][name])
            for (label, mode), _ in GRID.items():
                key = f"{label}_{mode}"
                numerator = np.sum([r["binned"][key][name]["transport"] for r in rows], axis=0)
                denominator = np.sum([r["binned"][key][name]["analytic"] for r in rows], axis=0)
                valid = denominator >= 1.0
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.plot(centers[valid], numerator[valid] / denominator[valid],
                        marker=".", label=f"{label} / {mode}")
            ax.set_xlabel(xlabel)
            ax.set_title(detector.upper())
            ax.axhline(1.0, color="grey", lw=0.6)
        axes[i, 0].set_ylabel("Transport / analytic PE")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Gate V · bins with at least 1 predicted PE across the holdouts")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "gatev")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = safe_output(args.out)
    collect(out, args.workers)
    plots(json.loads((out / "gatev_reconciliation.json").read_text()),
          ROOT / "figures/gatev_hk_sk_ratio.png")


if __name__ == "__main__":
    main()
