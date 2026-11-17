"""Stage 2 physics fits, calibrated closure, and Gate V ablations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import torch

from . import abs_render as ar
from . import lucid_bombs as lb
from . import r2_generator as gen
from . import r2_geometry as rg
from . import r2_store as rs
from .ext_r2_repair_probes import DETECTORS, EXTENSION, PROBE_ROOT, RETURNS, R_FID
from .ext_r2_repair_probes import dataset_path, safe_output

ROOT = RETURNS / EXTENSION
SEED = 20261117
VARIANTS = ("tof", "charge", "joint")
MU_BINS = np.array([0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, np.inf])
_WORKER = {}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False, default=float) + "\n")
    temp.replace(path)


def wall_distance(geometry, vertex):
    vertex = np.asarray(vertex)
    if geometry.shape == rg.SHAPE_SPHERE:
        return geometry.radius_m - np.linalg.norm(vertex, axis=-1)
    barrel = geometry.radius_m - np.linalg.norm(vertex[..., :2], axis=-1)
    cap = geometry.half_height_m - np.abs(vertex[..., 2])
    return np.minimum(barrel, cap)


def _init_worker(detector, yield_bounds):
    torch.set_num_threads(1)
    geometry = rg.load_r2_geometry(detector)
    cfg = gen.load_r2_config(*yield_bounds)
    _WORKER.update(geometry=geometry, cfg=cfg, detector=detector,
                   rule=gen.time_rule_for(geometry), models={})


def _fit(job):
    geometry, cfg = _WORKER["geometry"], _WORKER["cfg"]
    run, i, eid, truth, ln_y, hit, charge, time, expected, total_expected, replay, selected = job
    state = run.state(geometry)
    if i not in _WORKER["models"]:
        _WORKER["models"][i] = ar.PhotonBombRenderer(geometry, cfg, state)
    renderer = _WORKER["models"][i]
    ev = ar.EventData(hit, charge, time)
    theta = torch.tensor([*truth, ln_y, 0.0], dtype=renderer.dtype)
    t0 = ar.profile_t0(renderer, theta[:4], ev)
    theta[-1] = t0
    draws = None
    if selected:
        draw = lb.draw_event if _WORKER["detector"] == "wcte" else gen.draw_event
        draws = draw(geometry, cfg, run.run_id, eid, seed=run.seed, domain=lb.DOMAIN_PHOTON)
        if not np.allclose(truth, draws.vertex_m, atol=1e-9, rtol=0):
            raise ValueError(f"{eid}: generator replay vertex mismatch")
        if not np.isclose(ln_y, np.log(draws.yield_photons), atol=1e-9, rtol=0):
            raise ValueError(f"{eid}: generator replay light yield mismatch")

    dark_moved = None
    if replay and selected:
        rx = gen.readout(state, draws, geometry, cfg, _WORKER["rule"])
        if (not np.array_equal(hit, rx.pmt_id)
                or not np.allclose(charge, rx.charge_pe, rtol=1e-6, atol=1e-6)
                or not np.allclose(time, rx.time_ns, rtol=1e-6, atol=1e-6)):
            raise ValueError(f"{eid}: HK raw readout replay could not reproduce stored PMTs")
        dark_moved = rx.source_trigger_ns is not None and (
            rx.trigger_ns is None or abs(rx.trigger_ns - rx.source_trigger_ns) > 1e-9
        )
        t0 = draws.emission_offset_ns - (
            cfg.readout_window_ns[0] if rx.trigger_ns is None else rx.trigger_ns
        )
        theta[-1] = t0

    row = {"run_id": run.run_id, "event_id": eid, "selected": selected,
           "truth": truth.tolist(), "log_yield": ln_y, "dark_moved": dark_moved,
           "nearest_wall_m": float(wall_distance(geometry, truth)),
           "n_hit": len(hit), "hit_fraction": len(hit) / renderer.n_pmt,
           "total_charge": float(charge.sum()), "fits": {}}
    for variant in (VARIANTS if selected else ("joint",)):
        fit = ar.fit_reconstruction(renderer, ev, geometry, variant=variant, seed=SEED + eid)
        fit["error_m"] = (None if fit["theta"] is None else
                          float(np.linalg.norm(np.asarray(fit["theta"][:3]) - truth)))
        row["fits"][variant] = fit
    if not selected:
        return row

    row["crlb"] = ar.reconstruction_fisher(renderer, theta)
    with torch.no_grad():
        mean_charge, var_charge, p_hit, base, distance = renderer.moments(theta)
        replay_mu = base.numpy() * draws.qe_eff
        target_mu = expected if expected is not None else (
            lb.expected_charge(geometry, truth, draws.yield_photons, draws.qe_eff,
                               state.attenuation_length_m, cfg) * renderer.fixed_gain.numpy()
        )
        q = np.zeros(renderer.n_pmt)
        q[hit] = charge
        mean = mean_charge.numpy()
        variance = var_charge.numpy()
        residual = (time - renderer.predicted_time(theta, distance[hit]).numpy())
        row["time_residual_ns"] = residual.tolist()
        q_bin = np.clip(np.digitize(mean, MU_BINS) - 1, 0, len(MU_BINS) - 2)
        pulls = (q - mean) / np.sqrt(np.maximum(variance, 1e-12))
        bins = len(MU_BINS) - 1
        row["closure"] = {
            "mu_max_absolute_error_pe": float(np.max(np.abs(target_mu - replay_mu))),
            "total_mu_max_relative_error": float(
                abs(total_expected - replay_mu.sum()) / max(total_expected, 1e-12)),
            "n": np.bincount(q_bin, minlength=bins).tolist(),
            "pull_sum": np.bincount(q_bin, weights=pulls, minlength=bins).tolist(),
            "pull_sq_sum": np.bincount(q_bin, weights=pulls ** 2, minlength=bins).tolist(),
            "q_observed": np.bincount(q_bin, weights=q, minlength=bins).tolist(),
            "q_expected": np.bincount(q_bin, weights=mean, minlength=bins).tolist(),
        }
    return row


def _read_prediction(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _probe(detector, arm):
    return _read_prediction(PROBE_ROOT / f"by_arm/{detector}__{arm}__s9000.npz")


def _event_keys(rows):
    return list(zip(rows["run_ids"].astype(str).tolist(), rows["event_ids"].astype(int).tolist()))


def display_keys(detector):
    result = {}
    for arm, label in (("mae_mse", "mae"), ("sigreg_jepa", "fixed_jepa")):
        probe = _probe(detector, arm)
        errors = np.linalg.norm(probe["ridge_all_mean"] - probe["vertex"], axis=1)
        ordered = np.argsort(errors, kind="stable")
        keys = _event_keys(probe)
        result[f"worst_{label}"] = [keys[i] for i in ordered[-6:][::-1]]
        if label == "mae":
            result["median_mae"] = [keys[i] for i in ordered[len(ordered)//2 - 1:len(ordered)//2 + 2]]
    return result


def _photon_inputs(detector, n, *, display):
    cohort = rs.open_cohort(dataset_path(detector), "dev_probe_test")
    runs = gen.cohort_runs(cohort.name)
    manifests = [json.loads(raw) for raw in cohort.manifests]
    bounds = manifests[0].get("yield_range_photons")
    run_index = {run_id: i for i, run_id in enumerate(cohort.run_ids)}
    all_rows = []
    for index, run_id in enumerate(cohort.run_ids):
        data = cohort.run_truth(index)
        all_rows.extend((run_id, int(eid)) for eid in
                        data["event_ids"][data["domain"].astype(str) == lb.DOMAIN_PHOTON])
    rng = np.random.default_rng(SEED)
    selected = {all_rows[i] for i in rng.choice(len(all_rows), n, replace=False)}
    groups = display_keys(detector) if display else {}
    extra = {key for values in groups.values() for key in values}
    keys = sorted(selected | extra)
    by_run = defaultdict(list)
    for run_id, event_id in keys:
        by_run[run_id].append(event_id)

    with h5py.File(dataset_path(detector), "r") as f:
        expected = f[cohort.name]["truth"].get("expected_pe_pmt")
        jobs = []
        for run_id, events in by_run.items():
            index = run_index[run_id]
            run = runs[index]
            if run_id != run.run_id:
                raise ValueError(f"dataset and generator disagree on run index {index}")
            store, truth = cohort.run_store(index), cohort.run_truth(index)
            truth_idx = {int(e): k for k, e in enumerate(truth["event_ids"])}
            for eid in events:
                k = truth_idx[eid]
                sl = slice(store.offsets[k], store.offsets[k + 1])
                mu = expected[int(cohort.run_events[index]) + k] if expected is not None else None
                jobs.append((run, index, eid, truth["vertex_m"][k],
                             float(np.log(truth["yield_photons"][k])), store.pmt_id[sl],
                             store.charge_pe[sl], store.time_ns[sl], mu,
                             float(truth["expected_pe"][k]), detector == "hk",
                             (run_id, eid) in selected))
    return jobs, (None, None) if bounds is None else bounds, groups


def _summary(detector, rows):
    sample = [r for r in rows if r["selected"]]
    fits = {}
    for variant in VARIANTS:
        scores = [r["fits"][variant] for r in sample]
        errors = [x["error_m"] for x in scores if x["error_m"] is not None]
        fits[variant] = {"n_converged": sum(x["converged"] for x in scores),
                         "n_finite": len(errors), "mean_m": float(np.mean(errors)),
                         "median_m": float(np.median(errors)),
                         "mean_over_r_fid": float(np.mean(errors) / R_FID[detector])}
    bounds = [r["crlb"]["sigma_xyz_m"] for r in sample if r["crlb"]["ok"]]
    closure_rows = [r["closure"] for r in sample]
    stack = lambda key: np.sum([r[key] for r in closure_rows], axis=0)
    count = stack("n")
    pull_mean = stack("pull_sum") / np.maximum(count, 1)
    pull_rms = np.sqrt(stack("pull_sq_sum") / np.maximum(count, 1))
    inputs = {
        "n_events": len(sample), "r_fid_m": R_FID[detector], "fits": fits,
        "crlb": {"n_finite": len(bounds), "median_m": float(np.median(bounds)),
                 "model": "Expected information of Poisson count and hit-conditioned Gaussian arrival"},
        "closure": {
            "n_hit_charge_bins": count.tolist(), "pull_mean": pull_mean.tolist(),
            "pull_rms": pull_rms.tolist(), "mu_bins_pe": MU_BINS[:-1].tolist() + [None],
            "poisson_spe_pull_within_0p1": bool(np.all(abs(pull_rms[count > 100] - 1) <= 0.1)),
            "replay_total_mu_max_relative_error": max(r["total_mu_max_relative_error"]
                                                       for r in closure_rows),
            "replay_pmts_max_absolute_error_pe": max(r["mu_max_absolute_error_pe"]
                                                    for r in closure_rows),
            "n_stored_mu_events": 0 if detector == "wcte" else len(sample),
        },
    }
    if detector == "hk":
        reference = {}
        for arm in ("mae_mse", "sigreg_jepa", "target_posfree"):
            probe = _probe(detector, arm)
            reference[arm] = {
                form: dict(zip(_event_keys(probe),
                               np.linalg.norm(probe[form] - probe["vertex"], axis=1)))
                for form in ("ridge_all_mean", "ridge_spatial_moment_latent")
            }
        strata = {}
        for moved in (False, True):
            block = [r for r in sample if r["dark_moved"] == moved]
            keys = [(r["run_id"], r["event_id"]) for r in block]
            strata[str(moved).lower()] = {
                "n_events": len(block),
                "n_fraction": len(block) / len(sample),
                "mean_ml_errors_m": {v: float(np.mean([r["fits"][v]["error_m"] for r in block
                                                       if r["fits"][v]["error_m"] is not None]))
                                     for v in VARIANTS} if block else {},
                "mean_probe_errors_m": {arm: {form: float(np.mean([vals[key] for key in keys]))
                                               for form, vals in forms.items()}
                                        for arm, forms in reference.items()} if block else {},
            }
        inputs["dark_moved"] = strata
    return inputs


def physics(detector, out, n, workers, *, display):
    jobs, bounds, groups = _photon_inputs(detector, n, display=display)
    with mp.get_context("fork").Pool(workers, initializer=_init_worker,
                                     initargs=(detector, bounds)) as pool:
        rows = []
        for row in pool.imap_unordered(_fit, jobs, chunksize=1):
            rows.append(row)
            if len(rows) % 25 == 0:
                print(f"{detector} {len(rows)}/{len(jobs)}", flush=True)
    rows.sort(key=lambda r: (r["run_id"], r["event_id"]))
    summary = _summary(detector, rows)
    for row in rows:
        residual = row.pop("time_residual_ns", None)
        if residual is not None:
            # Keep a deterministic subsample for plots rather than multi-GB JSON.
            row["sample_time_residual_ns"] = residual[::max(1, len(residual)//80)]
    write_json(out / f"{detector}_physics.json",
               {"extension": EXTENSION, "summary": summary, "events": rows, "display": groups})
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True, choices=DETECTORS)
    parser.add_argument("--out", type=Path, default=ROOT / "diagnostics")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    physics(args.detector, safe_output(args.out), args.n, args.workers, display=not args.no_display)


if __name__ == "__main__":
    main()
