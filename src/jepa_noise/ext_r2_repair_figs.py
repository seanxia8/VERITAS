"""Matched-event figures and representation diagnostics for the repair results."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import socket

import numpy as np
import torch

from . import lucid_bombs as lb
from . import r2_generator as gen
from . import r2_geometry as rg
from . import r2_store as rs
from .ext_r2_repair_diagnostic_helpers import _cyl_coords, _display, _plt, _tank, linear_cka
from .ext_r2_repair_diagnostics import (
    DETECTORS, EXTENSION, ROOT, VARIANTS, _event_keys, _photon_inputs, _probe,
    wall_distance, write_json,
)
from .ext_r2_repair_probes import (
    ARMS, checkpoint_candidates, dataset_path, make_model, safe_output,
)
from .lucid_views import ViewMasker

RANDOM_STATE = 42
ARM_COLOR = {"init": "#555555", "mae_mse": "#0072B2", "sigreg_jepa": "#D55E00",
             "target_posfree": "#009E73", "mae_mse_rf": "#CC79A7"}


def _save(plt, fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145)
    plt.close(fig)


def _predictor_tables(detector):
    return {arm: dict(zip(_event_keys(data), zip(data["ridge_all_mean"],
                                              data["ridge_spatial_moment_latent"])))
            for arm in ("init", "mae_mse", "sigreg_jepa", "target_posfree")
            for data in [_probe(detector, arm)]}


def event_displays(detector, record, output):
    plt = _plt()
    from matplotlib.lines import Line2D
    geometry = rg.load_r2_geometry(detector)
    cohort = rs.open_cohort(dataset_path(detector), "dev_probe_test")
    indices = {r: i for i, r in enumerate(cohort.run_ids)}
    fit = {(r["run_id"], r["event_id"]): r for r in record["events"]}
    predictions = _predictor_tables(detector)
    projection = "mollweide" if detector == "juno" else None
    groups = record["display"]
    labels = ("worst_mae", "median_mae", "worst_fixed_jepa")
    colors = {"Truth": "#000000", "Joint ML": "#8A3FFC", **ARM_COLOR}
    for group in labels:
        rows = [(run_id, int(event_id)) for run_id, event_id in groups[group]]
        fig = plt.figure(figsize=(14, 2.9 * len(rows) + 0.8), layout="constrained")
        grid = fig.add_gridspec(len(rows), 4, width_ratios=(1.3, 1.3, 0.72, 0.72))
        for i, (run, eid) in enumerate(rows):
            row = fit[run, eid]
            store = cohort.run_store(indices[run])
            e = np.flatnonzero(store.event_ids == eid).item()
            sl = slice(store.offsets[e], store.offsets[e + 1])
            hit_id, q, t = store.pmt_id[sl], store.charge_pe[sl], store.time_ns[sl]
            hit = np.zeros(geometry.n_pmt, bool)
            hit[hit_id] = True
            charge, time = np.full(geometry.n_pmt, np.nan), np.full(geometry.n_pmt, np.nan)
            charge[hit_id] = np.log(np.maximum(q, 1e-6))
            time[hit_id] = t
            ax_q = fig.add_subplot(grid[i, 0], projection=projection)
            ax_t = fig.add_subplot(grid[i, 1], projection=projection)
            sc = _display(ax_q, geometry, charge, hit, "Log charge", "viridis")
            fig.colorbar(sc, ax=ax_q, shrink=0.5)
            sc = _display(ax_t, geometry, time, hit, "First-hit time (ns)", "cividis")
            fig.colorbar(sc, ax=ax_t, shrink=0.5)
            dark = hit_id[t < 0]
            if len(dark):
                p = geometry.positions_m
                if projection:
                    x = np.arctan2(p[:, 1], p[:, 0])
                    y = np.arcsin(np.clip(p[:, 2] / np.linalg.norm(p, axis=1), -1, 1))
                else:
                    x, y = _cyl_coords(geometry, p)
                ax_t.scatter(x[dark], y[dark], marker="x", c="#E63946", s=8, lw=0.5)
            truth = np.asarray(row["truth"])
            reconstructed = [("Truth", truth, "*")]
            if row["fits"]["joint"]["theta"] is not None:
                reconstructed.append(("Joint ML", row["fits"]["joint"]["theta"][:3], "P"))
            for arm, by_event in predictions.items():
                baseline, spatial = by_event[run, eid]
                reconstructed.append((arm, baseline, "o"))
                if arm in ("mae_mse", "target_posfree"):
                    reconstructed.append((arm, spatial, "s"))
            for j, view in enumerate(("top", "side")):
                ax = fig.add_subplot(grid[i, j + 2])
                _tank(ax, geometry, view)
                if view == "side":
                    ax.set_xlim(0.0, 1.05 * geometry.radius_m)
                for label, point, marker in reconstructed:
                    x, y, z = point
                    xy = (x, y) if view == "top" else (np.hypot(x, y), z)
                    ax.scatter(*xy, marker=marker, s=33 if marker == "*" else 12,
                               color=colors[label], zorder=3)
                ax.set(xlabel="x (m)" if view == "top" else "r (m)",
                       ylabel="y (m)" if view == "top" else "z (m)")
            base_error = np.linalg.norm(predictions["mae_mse"][run, eid][0] - truth)
            errors = [f"{arm} {np.linalg.norm(table[run, eid][0] - truth):.2f}"
                      for arm, table in predictions.items()]
            ax_t.set_title("First-hit time (ns)\nBaseline errors (m) " + " · ".join(errors))
            ax_q.set_title(f"Event {eid} · ln Y {row['log_yield']:.2f} · {len(q):,} hits\n"
                           f"{q.sum():.0f} PE · wall {row['nearest_wall_m']:.3f} m · "
                           f"MAE baseline error {base_error:.3f} m")
        legend = [Line2D([], [], marker=m, ls="", color=colors[label], label=label)
                  for label, m in (("Truth", "*"), ("Joint ML", "P"),
                                   *[(arm, "o") for arm in predictions])]
        legend.append(Line2D([], [], marker="s", ls="", color="0.3", label="Spatial-moment Ridge"))
        legend.append(Line2D([], [], marker="x", ls="", color="#E63946", label="Time below 0 ns"))
        fig.legend(handles=legend, loc="outside lower center", ncols=8)
        fig.suptitle(f"{detector.upper()} · {group.replace('_', ' ')}")
        _save(plt, fig, output / "event_displays" / f"{detector}_{group}.png")


def _binned_axes(ax, values, series, *, bins=10):
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    assignment = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, bins - 1)
    for label, error in series.items():
        candidates = [i for i in range(bins) if np.any((assignment == i) & np.isfinite(error))]
        x = [np.median(values[assignment == i]) for i in candidates]
        y = [np.nanmedian(error[assignment == i]) for i in candidates]
        ax.plot(x, y, marker=".", lw=1, label=label)
    ax.set_ylabel("Median error (m)")
    ax.grid(alpha=0.2)


def error_anatomy(detector, record, output):
    plt = _plt()
    data = [row for row in record["events"] if row["selected"]]
    run_ids = np.asarray([row["run_id"] for row in data])
    geometry = rg.load_r2_geometry(detector)
    vertex = np.asarray([row["truth"] for row in data])
    find = _predictor_tables(detector)
    key = [(row["run_id"], row["event_id"]) for row in data]
    series = {"Joint ML": np.asarray([row["fits"]["joint"]["error_m"] for row in data], float)}
    for arm in ("mae_mse", "sigreg_jepa", "target_posfree"):
        vertices = np.asarray([find[arm][k] for k in key])
        errors = np.linalg.norm(vertices - vertex[:, None, :], axis=-1)
        series[f"{arm} · all-mean"] = errors[:, 0]
        series[f"{arm} · spatial"] = errors[:, 1]
    indices = np.searchsorted(np.unique(run_ids), run_ids)
    axes_names = ("True radius (m)", "|z| (m)", "Distance to wall (m)", "ln Y", "Run index")
    values = (np.linalg.norm(vertex[:, :2], axis=1), np.abs(vertex[:, 2]),
              wall_distance(geometry, vertex), np.asarray([row["log_yield"] for row in data]), indices)
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.5), layout="constrained")
    for panel, x, name in zip(ax, values, axes_names):
        _binned_axes(panel, x, series)
        panel.set_xlabel(name)
        panel.set_yscale("log")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncols=4)
    fig.suptitle(f"{detector.upper()} · matched 500-event photon-bomb sample")
    _save(plt, fig, output / "error_anatomy" / f"{detector}.png")


def physics_figures(records, output):
    plt = _plt()
    fig, axes = plt.subplots(4, 2, figsize=(9, 12), layout="constrained")
    sanity, sim = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for i, detector in enumerate(DETECTORS):
        sample = [row for row in records[detector]["events"] if row["selected"]]
        ly = np.asarray([row["log_yield"] for row in sample])
        wall = np.asarray([row["nearest_wall_m"] for row in sample])
        errors = {v: np.asarray([row["fits"][v]["error_m"] for row in sample], float)
                  for v in VARIANTS}
        errors["CRLB"] = np.asarray([row["crlb"]["sigma_xyz_m"] for row in sample], float)
        for ax, values, label in zip(axes[i], (ly, wall), ("ln Y", "Distance to wall (m)")):
            _binned_axes(ax, values, errors)
            ax.set_ylabel("Median error or bound (m)")
            ax.set_title(detector.upper())
            ax.set_xlabel(label)
            ax.set_yscale("log")
        y = np.exp(ly)
        charge = np.asarray([row["total_charge"] for row in sample])
        hit = np.asarray([row["hit_fraction"] for row in sample])
        residual = np.concatenate([row["sample_time_residual_ns"] for row in sample])
        sim[0].scatter(y, charge, s=4, alpha=0.35, label=detector.upper())
        sim[1].scatter(y, hit, s=4, alpha=0.35, label=detector.upper())
        sim[2].hist(residual, bins=np.linspace(-15, 15, 61), histtype="step", density=True,
                    label=detector.upper())
        scale = {"total_charge_slope": float(np.polyfit(ly, np.log(charge), 1)[0]),
                 "hit_fraction_slope": float(np.polyfit(ly, np.log(hit), 1)[0]),
                 "median_time_residual_ns": float(np.median(residual)),
                 "central_68_time_width_ns": float(np.diff(np.quantile(residual, (0.16, 0.84)))[0] / 2)}
        records[detector]["summary"]["simulation_sanity"] = scale
    axes[0, 0].legend()
    sim[0].set(xscale="log", yscale="log", xlabel="Photon yield", ylabel="Detected charge (PE)")
    sim[1].set(xscale="log", yscale="log", xlabel="Photon yield", ylabel="Hit fraction")
    sim[2].set(xlabel="Time residual (ns)", ylabel="Density")
    sim[2].legend()
    _save(plt, fig, output / "physics_ceiling_vs_yield.png")
    _save(plt, sanity, output / "sim_sanity_summary.png")


def generator_closure(detector, output):
    """Compare measured sparse charge with the generator expectation on the 500-event panel."""
    geometry = rg.load_r2_geometry(detector)
    jobs, bounds, _ = _photon_inputs(detector, 500, display=False)
    cfg = gen.load_r2_config(*bounds)
    fixed_gain = lb.fixed_charge_gain(geometry, lb.generator_reference(cfg))
    dark = cfg.dark_rate_hz * (cfg.readout_window_ns[1] - cfg.readout_window_ns[0]) * 1e-9
    edges = np.linspace(0, rg.longest_path_m(geometry), 21), np.linspace(0, 90, 19)
    counts = [np.zeros(len(x) - 1) for x in edges]
    model = [np.zeros_like(a) for a in counts]
    observed = [np.zeros_like(a) for a in counts]
    states = {}
    for run, _, eid, truth, _, hit, charge, _, expected, _, _, _ in jobs:
        if run.run_id not in states:
            states[run.run_id] = run.state(geometry)
        state = states[run.run_id]
        if expected is None:
            draw = lb.draw_event(geometry, cfg, run.run_id, eid, seed=run.seed,
                                 domain=lb.DOMAIN_PHOTON)
            expected = (lb.expected_charge(geometry, truth, draw.yield_photons, draw.qe_eff,
                                           state.attenuation_length_m, cfg) * fixed_gain)
        expected_charge = (expected + dark) * state.gains
        delta = geometry.positions_m - truth
        distance = np.linalg.norm(delta, axis=1)
        cosine = np.abs(np.einsum("ij,ij->i", delta, geometry.directions)) / distance
        incidence = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
        for j, values in enumerate((distance, incidence)):
            counts[j] += np.histogram(values, bins=edges[j])[0]
            model[j] += np.histogram(values, bins=edges[j], weights=expected_charge)[0]
            observed[j] += np.histogram(values[hit], bins=edges[j], weights=charge)[0]
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for j, title in enumerate(("Vertex to PMT distance (m)", "PMT incidence angle (deg)")):
        center = 0.5 * (edges[j][:-1] + edges[j][1:])
        valid = counts[j] > 0
        for label, values in (("Measured prompt charge", observed[j]), ("Generator mean", model[j])):
            axes[j].plot(center[valid], values[valid] / counts[j][valid], marker=".", label=label)
        axes[j].set(xlabel=title, ylabel="Mean charge per PMT (PE)", yscale="log")
    axes[0].legend()
    fig.suptitle(f"{detector.upper()} · photon-bomb generator closure")
    _save(plt, fig, output / f"simulation_closure_{detector}.png")
    return {axis: {"edges": edges[j].tolist(), "counts": counts[j].tolist(),
                   "generator_charge_sum": model[j].tolist(),
                   "observed_prompt_charge_sum": observed[j].tolist()}
            for j, axis in enumerate(("distance_m", "incidence_deg"))}


def _project(x, method):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    if method == "pca":
        return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(x)
    if method == "tsne":
        return TSNE(n_components=2, perplexity=30, init="pca", random_state=RANDOM_STATE,
                    learning_rate="auto", n_jobs=1).fit_transform(x)
    return umap.UMAP(n_components=2, n_neighbors=15, random_state=RANDOM_STATE,
                     n_jobs=1).fit_transform(x)


def _map_grid(plt, projections, colors, output):
    fig, axes = plt.subplots(len(projections), len(colors), squeeze=False,
                             figsize=(3.0 * len(colors), 2.4 * len(projections)), layout="constrained")
    for i, (arm, embedding) in enumerate(projections.items()):
        for j, (title, values) in enumerate(colors[arm].items()):
            ax = axes[i, j]
            c = ax.scatter(*embedding.T, c=values, s=6, cmap="viridis", lw=0)
            ax.set_title(f"{arm} · {title}")
            ax.set(xticks=[], yticks=[])
            fig.colorbar(c, ax=ax, shrink=0.58)
    _save(plt, fig, output)


@torch.inference_mode()
def token_variance(detector, arm, keys, *, cohort, geometry, device):
    rule = gen.time_rule_for(geometry)
    scaling = rs.scaling_for(geometry, rule.time_scale_ns)
    model, identity = make_model(arm, 9000, geometry, scaling,
                                 checkpoint_candidates(detector, arm, 9000), device)
    masker = ViewMasker(geometry, "within", n_min=0, seed=0)
    run_to_index = {r: i for i, r in enumerate(cohort.run_ids)}
    counts = np.zeros((2, geometry.n_pmt), dtype=np.int64)
    sums = np.zeros((2, geometry.n_pmt, model.d_model))
    norms = np.zeros_like(counts, dtype=float)
    by_run = {}
    for run_id, eid in keys:
        by_run.setdefault(run_id, []).append(eid)
    representative = keys[0]
    token_example = None
    for run_id, events in by_run.items():
        store = cohort.run_store(run_to_index[run_id])
        for start in range(0, len(events), 8):
            chunk = events[start:start + 8]
            b = rs.tokenise_sparse(store, geometry, event_ids=chunk, scaling=scaling, masker=masker)
            tokens = model.jepa.encode(b.features.to(device)).float().cpu().numpy()
            for event, eid in enumerate(b.event_ids):
                hit = b.hit_flags[event]
                z = tokens[event].astype(float)
                length2 = np.einsum("nd,nd->n", z, z)
                for g, mask in enumerate((~hit, hit)):
                    counts[g, mask] += 1
                    sums[g, mask] += z[mask]
                    norms[g, mask] += length2[mask]
                if (run_id, eid) == representative:
                    token_example = (tokens[event], b.log_charge[event], hit.copy())
    event_count = len(keys)
    if not np.all(counts.sum(axis=0) == event_count):
        raise ValueError(f"{detector}/{arm}: incomplete same-PMT variance observations")
    means = sums / np.maximum(counts[..., None], 1)
    pmt_mean = sums.sum(axis=0) / event_count
    fractions = counts / event_count
    within = (norms - counts * np.einsum("gnd,gnd->gn", means, means)) / event_count
    between = (fractions * ((means - pmt_mean) ** 2).sum(axis=2)).sum(axis=0)
    per_pmt = {"hit_frequency": fractions[1], "hit_vs_unhit": between,
               "within_hit": within[1], "within_unhit": within[0],
               "same_pmt_across_event_variance": between + within.sum(axis=0)}
    result = {"n_events": event_count, "n_pmts": geometry.n_pmt,
              "static_layout": float(((pmt_mean - pmt_mean.mean(axis=0)) ** 2).sum(axis=1).mean()),
              "hit_vs_unhit": float(between.mean()), "within_hit": float(within[1].mean()),
              "within_unhit": float(within[0].mean()),
              "same_pmt_across_event_variance": float(per_pmt["same_pmt_across_event_variance"].mean()),
              "checkpoint": identity["checkpoint"], "checkpoint_sha256": identity["checkpoint_sha256"]}
    result["total_variance"] = result["static_layout"] + result["same_pmt_across_event_variance"]
    result["component_fractions"] = {
        k: result[k] / result["total_variance"]
        for k in ("static_layout", "hit_vs_unhit", "within_hit", "within_unhit")
    }
    return result, per_pmt, token_example


def latent(detector, out, device):
    plt = _plt()
    geometry = rg.load_r2_geometry(detector)
    cohort = rs.open_cohort(dataset_path(detector), "dev_probe_test")
    data = {arm: _probe(detector, arm) for arm in ARMS if arm != "mae_mse_rf" or
            checkpoint_candidates(detector, arm, 9000)}
    keys = _event_keys(data["init"])
    if any(_event_keys(item) != keys for item in data.values()):
        raise ValueError(f"{detector}: latent embeddings use different event ordering")
    rng = np.random.default_rng(RANDOM_STATE)
    take = rng.choice(len(keys), 2000, replace=False)
    token_keys = [keys[index] for index in take[:200]]
    pmt_take = np.sort(rng.choice(geometry.n_pmt, min(2000, geometry.n_pmt), replace=False))
    summary = {"n_event_maps": len(take), "event_sample_seed": RANDOM_STATE,
               "representative_event": token_keys[0], "n_plotted_token_pmts": len(pmt_take),
               "host": socket.gethostname(), "variance": {}, "cka": {}}
    form_labels = ("all_mean", "hit_mean")
    vertices = data["init"]["vertex"][take]
    runs = data["init"]["run_ids"][take]
    color = {"r (m)": np.linalg.norm(vertices[:, :2], axis=1), "z (m)": vertices[:, 2],
             "ln Y": data["init"]["log_yield"][take], "run": np.searchsorted(np.unique(runs), runs)}
    for form in form_labels:
        mats = {arm: values[form][take] for arm, values in data.items()}
        cka = {left: {right: (1.0 if left == right else None) for right in mats} for left in mats}
        for left, right in combinations(mats, 2):
            cka[left][right] = cka[right][left] = linear_cka(mats[left], mats[right])
        summary["cka"][form] = cka
        for method in ("pca", "tsne", "umap"):
            layout = {arm: _project(matrix, method) for arm, matrix in mats.items()}
            _map_grid(plt, layout, {arm: color for arm in layout},
                      out / f"{detector}_{form}_{method}.png")
    token_cloud, token_colors = {}, {}
    pos = geometry.positions_m[pmt_take]
    out.mkdir(parents=True, exist_ok=True)
    for arm in data:
        result, per_pmt, example = token_variance(
            detector, arm, token_keys, cohort=cohort, geometry=geometry, device=device)
        summary["variance"][arm] = result
        variance_path = out / f"{detector}_{arm}_per_pmt_variance.npz"
        np.savez_compressed(variance_path, pmt_index=np.arange(geometry.n_pmt), **per_pmt)
        z, charge, hit = example
        token_cloud[arm] = z[pmt_take]
        token_colors[arm] = {"PMT azimuth": np.arctan2(pos[:, 1], pos[:, 0]),
                             "PMT height": pos[:, 2],
                             "Scaled log-charge": np.where(hit[pmt_take], charge[pmt_take], -0.2)}
    for method in ("tsne", "umap"):
        layout = {a: _project(x, method) for a, x in token_cloud.items()}
        _map_grid(plt, layout, token_colors, out / f"{detector}_tokens_{method}.png")
    write_json(ROOT / "diagnostics" / f"{detector}_latent.json", summary)


def collect_physics(figures):
    records = {detector: json.loads((ROOT / "diagnostics" / f"{detector}_physics.json").read_text())
               for detector in DETECTORS}
    for detector, record in records.items():
        if record["summary"]["n_events"] != 500:
            raise ValueError(f"{detector}: refusing to aggregate incomplete physics sample")
        event_displays(detector, record, figures)
        error_anatomy(detector, record, figures)
        record["summary"]["closure_by_pmt_geometry"] = generator_closure(detector, figures)
        source = ROOT / "diagnostics" / f"{detector}_physics.json"
        record["summary"]["per_event_artifact"] = {"path": str(source), "sha256": rs.sha256_file(source)}
    physics_figures(records, figures)
    write_json(ROOT / "diagnostics" / "physics_ceiling.json",
               {"extension": EXTENSION, "sample": "500 held-out photon bombs per detector",
                "detectors": {det: row["summary"] for det, row in records.items()}})


def collect_latent():
    latent = {detector: json.loads((ROOT / "diagnostics" / f"{detector}_latent.json").read_text())
              for detector in DETECTORS}
    for detector, summary in latent.items():
        for arm, variance in summary["variance"].items():
            relative = Path("figures/latent") / f"{detector}_{arm}_per_pmt_variance.npz"
            variance["per_pmt_artifact"] = {
                "path_from_repair_root": str(relative), "sha256": rs.sha256_file(ROOT / relative)
            }
    write_json(ROOT / "diagnostics" / "latent_variance_cka.json",
               {"extension": EXTENSION, "detectors": latent})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=DETECTORS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--collect", choices=("physics", "latent", "all"), default="all")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    out = safe_output(args.out)
    if args.detector:
        latent(args.detector, out / "latent", args.device)
    else:
        if args.collect in ("physics", "all"):
            collect_physics(out)
        if args.collect in ("latent", "all"):
            collect_latent()


if __name__ == "__main__":
    main()
