"""Evaluate repaired readouts on held-out runs of frozen Round-1/2 checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import ext_r2 as r2
from . import lucid_bombs as lb
from . import r2_generator as gen
from . import r2_geometry as rg
from . import r2_store as rs
from .lucid_views import ViewMasker
from .r2_probe_repair import fit_mlp, fit_ridge, moment_features, pool_tokens

EXTENSION = "r2_repair_2026-11-17"
RETURNS = Path("/ceph/dwong/jepa/returns")
R2 = RETURNS / "r2_multidetector_2026-11-11"
PROBE_ROOT = RETURNS / EXTENSION / "probes"
DETECTORS = ("wcte", "sk", "juno", "hk")
ARMS = ("init", "mae_mse", "mae_mse_rf", "sigreg_jepa", "target_posfree")
SEEDS = (9000, 9001)
R_FID = dict(zip(DETECTORS, (1.262, 13.570, 14.000, 25.944)))
VERTEX_FORMS = (
    "all_mean", "hit_mean", "weighted_means", "spatial_moment_latent",
    "spatial_moment_full", "geometry_summary", "raw_spatial_moment",
)
TOKEN_FORMS = ("raw_unmasked_r2", "post_ln_unmasked_r2", "masked_target_pred_r2")
XYZ = ("x", "y", "z")


def dataset_path(detector, kind=None):
    """Probe pack; LUCiD data by default on lucid-mc (``JEPA_DATA_KIND=analytic_toy`` for the toy)."""
    from . import data_roots as dr
    return dr.pack(detector, kind)


def checkpoint_candidates(detector, arm, seed):
    """List checkpoints in the frozen campaign namespaces, including partial copies."""
    if arm == "init":
        return []
    if detector == "wcte":
        candidates = {
            "mae_mse": RETURNS / "ext_mae_dejitter_20261031/train/runs"
            / f"ext_dev_s005_within_trigger__mae_mse__s{seed}/final.pt",
            "sigreg_jepa": Path(f"/work/dwong/jepa_a10_work/fixed_s{seed}/runs")
            / f"dev_s005_within__sigreg_jepa__s{seed}/final.pt",
            "target_posfree": RETURNS / "ext_shortcut_20261107/train/runs"
            / f"dev_s005_within__sigreg_jepa__s{seed}/final.pt",
        }
        paths = [candidates[arm]] if arm in candidates else []
    else:
        tag = f"{detector}__{arm}__s{seed}/checkpoint.pt"
        paths = [R2 / "runs" / tag, R2 / "batch/runs" / tag]
    return [path for path in paths if path.is_file()]


def final_checkpoint(paths, arm, seed):
    """Select a final matching checkpoint and catch ambiguous complete copies."""
    valid, seen = [], []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        identity = payload.get("identity", {})
        step = int(payload.get("step", -1))
        seen.append((str(path), step, identity.get("arm"), identity.get("seed")))
        if (step == r2.STEPS and identity.get("seed") == seed
                and identity.get("arm") in (arm, r2.base_arm_name(arm))):
            valid.append((path, payload, rs.sha256_file(path)))
    if not valid:
        raise ValueError(f"no matching final checkpoint for {arm}/{seed}: {seen}")
    if len({digest for _, _, digest in valid}) > 1:
        raise ValueError(f"conflicting final copies for {arm}/{seed}: {seen}")
    return valid[0]


def make_model(arm, seed, geometry, scaling, paths, device):
    model = r2.build_r2_arm(arm, geometry, seed, scaling=scaling)
    meta = {"checkpoint": None, "checkpoint_sha256": None, "step": 0}
    if arm != "init":
        path, payload, digest = final_checkpoint(paths, arm, seed)
        model.load_state_dict(payload["model"])
        meta = {"checkpoint": str(path), "checkpoint_sha256": digest,
                "step": int(payload["step"]), "identity": payload["identity"]}
    return model.to(device).eval(), meta


@torch.inference_mode()
def stream(model, geometry, scaling, cohort, *, device, chunk_size, events_per_run=None):
    """Keep event features and matched masked hit samples instead of full token matrices."""
    rng = np.random.default_rng(20261117)
    features = {name: [] for name in VERTEX_FORMS + TOKEN_FORMS}
    keys = ("event_ids", "run_ids", "vertex", "log_yield", "token_y", "token_runs")
    rows = {key: [] for key in keys}
    masker = ViewMasker(geometry, "within", seed=0, n_min=0)
    for i, run_id in enumerate(cohort.run_ids):
        truth = cohort.run_truth(i)
        photon = np.flatnonzero(truth["domain"].astype(str) == lb.DOMAIN_PHOTON)
        store = cohort.run_store(i)
        if events_per_run is not None:
            photon = photon[:events_per_run]
        for start in range(0, len(photon), chunk_size):
            subset = photon[start:start + chunk_size]
            ids = truth["event_ids"][subset]
            batch = rs.tokenise_sparse(store, geometry, event_ids=ids, scaling=scaling,
                                       masker=masker)
            if not np.array_equal(ids, batch.event_ids):
                raise RuntimeError(f"evaluation lost alignment at {run_id}")
            x = batch.features
            encoded = model.encode_tokens(x.to(device))
            z = encoded.float().cpu().numpy()
            raw = x.numpy()
            latent, full = moment_features(z, raw)
            _, raw_full = moment_features(raw[..., 6:9], raw)
            features["all_mean"].append(pool_tokens(z, raw, "all_mean"))
            features["hit_mean"].append(latent[:, :z.shape[-1]])
            features["weighted_means"].append(latent[:, :3 * z.shape[-1]])
            features["spatial_moment_latent"].append(latent)
            features["spatial_moment_full"].append(full)
            features["geometry_summary"].append(full[:, -15:])
            features["raw_spatial_moment"].append(raw_full)
            rows["event_ids"].append(ids)
            rows["run_ids"].append(np.repeat(run_id, len(ids)))
            rows["vertex"].append(truth["vertex_m"][subset])
            rows["log_yield"].append(np.log(truth["yield_photons"][subset]))

            # Identical hit targets for the three token controls. Feature columns
            # are hidden from the context encoder when producing predictions.
            x_dev = x.to(device)
            hidden = batch.hidden.to(device)
            target = model._hidden_index(batch).numpy()
            sample_rows, columns = [], []
            for event, indices in enumerate(target):
                hits = np.flatnonzero(raw[event, indices, 8] > 0.5)
                if len(hits) > 32:
                    hits = np.sort(rng.choice(hits, size=32, replace=False))
                sample_rows.extend([event] * len(hits))
                columns.extend(hits)
            if not columns:
                continue
            sample_rows = np.asarray(sample_rows)
            columns = np.asarray(columns)
            pmts = target[sample_rows, columns]
            context = model.jepa.encode(model._encoder_features(x_dev), hidden_mask=hidden)
            prediction = model.jepa.predict(context, batch.context_index.to(device),
                                            torch.as_tensor(target, device=device))
            unmasked = encoded[sample_rows, pmts].float()
            features["raw_unmasked_r2"].append(unmasked.cpu().numpy())
            features["post_ln_unmasked_r2"].append(F.layer_norm(unmasked, (z.shape[-1],))
                                                   .cpu().numpy())
            features["masked_target_pred_r2"].append(prediction[sample_rows, columns]
                                                     .float().cpu().numpy())
            rows["token_y"].append(raw[sample_rows, pmts, 6:8])
            rows["token_runs"].append(np.repeat(run_id, len(columns)))
    return {key: np.concatenate(value) for key, value in (features | rows).items()}


def score_vertex(prediction, truth, radius):
    errors = np.linalg.norm(prediction - truth, axis=-1)
    if not np.isfinite(errors).all():
        raise RuntimeError("vertex probe predictions are incomplete or non-finite")
    return {"n": len(errors), "mean_m": float(errors.mean()),
            "median_m": float(np.median(errors)), "p68_m": float(np.quantile(errors, 0.68)),
            "mean_over_r_fid": float(errors.mean() / radius)}


def evaluate(model, detector, seed, geometry, scaling, cohorts, *, device, chunk_size,
             events_per_run=None):
    train, test = [stream(model, geometry, scaling, cohort, device=device,
                          chunk_size=chunk_size, events_per_run=events_per_run)
                   for cohort in cohorts]
    if np.intersect1d(train["run_ids"], test["run_ids"]).size:
        raise ValueError("probe-train and probe-test run IDs overlap")
    vertex, arrays = {}, {}
    radius = R_FID[detector]
    for form in VERTEX_FORMS:
        prediction, alpha = fit_ridge(train[form], train["vertex"], train["run_ids"], test[form],
                                      form=form, target_names=XYZ)
        key = f"ridge_{form}"
        vertex[key] = dict(score_vertex(prediction, test["vertex"], radius), alpha=alpha)
        arrays[key] = prediction
    for form in ("hit_mean", "spatial_moment_full"):
        prediction, validation = fit_mlp(train[form], train["vertex"], train["run_ids"],
                                         test[form], seed=seed, device=device)
        key = "mlp_spatial_moment" if form == "spatial_moment_full" else "mlp_hit_mean"
        vertex[key] = dict(score_vertex(prediction, test["vertex"], radius), **validation)
        arrays[key] = prediction
    token = {}
    for form in TOKEN_FORMS:
        names = ("token_log_charge_r2", "token_time_r2")
        prediction, alpha = fit_ridge(train[form], train["token_y"], train["token_runs"],
                                      test[form], form=form, target_names=names)
        truth = test["token_y"]
        sse = ((prediction - truth) ** 2).sum(axis=0)
        sst = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)
        token[form] = {name: float(1.0 - err / total) if total > 0 and np.isfinite(err)
                       else None for name, err, total in zip(names, sse, sst)}
        token[form].update(alpha=alpha, n_train=len(train["token_y"]), n_test=len(truth))
    for key in ("all_mean", "hit_mean", "vertex", "log_yield", "event_ids", "run_ids"):
        arrays[key] = test[key]
    return {"vertex": vertex, "token": token, "r_fid_m": radius,
            "n_probe_train": len(train["vertex"]), "n_probe_test": len(test["vertex"])}, arrays


def safe_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(RETURNS / EXTENSION):
        raise ValueError(f"remote outputs must stay under {RETURNS / EXTENSION}")
    return path


def probe_matrix(args):
    out = safe_output(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    detector = args.detector
    geometry = rg.load_r2_geometry(detector)
    rule = gen.time_rule_for(geometry)
    scaling = rs.scaling_for(geometry, rule.time_scale_ns)
    path = dataset_path(detector)
    cohorts = [rs.open_cohort(path, f"dev_probe_{split}") for split in ("train", "test")]
    for arm in args.arms:
        for seed in args.seeds:
            name = f"{detector}__{arm}__s{seed}"
            record = out / f"{name}.json"
            if record.exists():
                continue
            candidates = checkpoint_candidates(detector, arm, seed)
            if not candidates and arm == "mae_mse_rf":
                print(f"{name}: optional checkpoint unavailable", flush=True)
                continue
            start = time.monotonic()
            model, meta = make_model(arm, seed, geometry, scaling, candidates, device)
            print(f"{name}: {meta['checkpoint'] or 'random initialization'}", flush=True)
            scores, arrays = evaluate(model, detector, seed, geometry, scaling, cohorts,
                                      device=device, chunk_size=args.chunk_size,
                                      events_per_run=args.events_per_run)
            row = {"extension": EXTENSION, "registered": False, "detector": detector, "arm": arm,
                   "seed": seed, "device": str(device), "events_per_run": args.events_per_run,
                   "hdf5": str(path), "wall_seconds": time.monotonic() - start, **meta, **scores}
            np.savez_compressed(out / f"{name}.npz", **arrays)
            temporary = record.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n")
            temporary.replace(record)
            print(f"{name}: {row['wall_seconds']:.1f} s", flush=True)


def aggregate(root, out):
    rows, unavailable = [], []
    lines = ["# Frozen-checkpoint probe repair", "",
             "Mean 3-D error in metres / fraction of the fiducial radius. "
             "Both fit and model selection use complete probe-training runs.", "",
             "| Detector | Arm | Seed | All-mean Ridge | Hit + weighted means Ridge | "
             "Spatial latent Ridge | Spatial full Ridge | Spatial MLP |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for detector in DETECTORS:
        for seed in SEEDS:
            for arm in ARMS:
                path = root / f"{detector}__{arm}__s{seed}.json"
                if not path.exists() and arm == "mae_mse_rf":
                    if not checkpoint_candidates(detector, arm, seed):
                        unavailable.append({"detector": detector, "arm": arm, "seed": seed})
                        continue
                row = json.loads(path.read_text())
                if (row["detector"], row["arm"], row["seed"]) != (detector, arm, seed):
                    raise ValueError(f"row identity differs from its path: {path}")
                if not path.with_suffix(".npz").is_file():
                    raise FileNotFoundError(f"missing per-event predictions for {path}")
                if row["events_per_run"] is not None:
                    raise ValueError(f"refusing to aggregate smoke output {path}")
                rows.append(row)
    init = {(row["detector"], row["seed"]): row for row in rows if row["arm"] == "init"}
    for row in rows:
        control = init[row["detector"], row["seed"]]["token"]["post_ln_unmasked_r2"]
        scores = row["token"]["post_ln_unmasked_r2"]
        scores["delta_vs_init"] = {
            key: scores[key] - control[key] if scores[key] is not None and control[key] is not None
            else None for key in ("token_log_charge_r2", "token_time_r2")
        }
        cells = [row["detector"], row["arm"], str(row["seed"])]
        for key in ("ridge_all_mean", "ridge_weighted_means", "ridge_spatial_moment_latent",
                    "ridge_spatial_moment_full", "mlp_spatial_moment"):
            result = row["vertex"][key]
            cells.append(f"{result['mean_m']:.4f} / {result['mean_over_r_fid']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "Token R² uses identical hit target indices for the unmasked, post-LayerNorm "
                  "and masked prediction controls. Per-event probe predictions and checkpoint "
                  "provenance are keyed by detector, run and event on Ceph.", ""])
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_repair_results.json").write_text(json.dumps(
        {"extension": EXTENSION, "registered": False, "runs": rows,
         "unavailable_optional_checkpoints": unavailable},
        indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "PROBE_REPAIR_TABLE.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--detector", required=True, choices=DETECTORS)
    probe.add_argument("--arms", choices=ARMS, nargs="+", default=ARMS)
    probe.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    probe.add_argument("--device", default="cpu")
    probe.add_argument("--memory-fraction", type=float, default=0.45)
    probe.add_argument("--chunk-size", type=int, default=8)
    probe.add_argument("--events-per-run", type=int)
    probe.add_argument("--out", type=Path, default=PROBE_ROOT / "by_arm")
    collect = sub.add_parser("aggregate")
    collect.add_argument("--root", type=Path, default=PROBE_ROOT / "by_arm")
    collect.add_argument("--out", type=Path, default=PROBE_ROOT)
    args = parser.parse_args()
    if args.command == "probe":
        if args.device.startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction, args.device)
        probe_matrix(args)
    else:
        aggregate(args.root, safe_output(args.out))


if __name__ == "__main__":
    main()
