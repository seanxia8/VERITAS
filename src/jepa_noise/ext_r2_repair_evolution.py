"""Evaluate all Stage 5 checkpoints and generate the frozen-step latent analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import ext_r2 as r2
from . import ext_r2_repair_probes as probes
from . import ext_readout as er
from . import lucid_bombs as lb
from . import r2_store as rs
from . import data_roots as dr
from .ext_r2_repair_train import (
    LUCID_EXTENSION, LUCID_MATRIX, MATRIX, RETURNS, ROOT, SEED, SNAPSHOTS, geometry_and_scaling,
    run_id, write_json,
)


def campaign_root(kind):
    """Output root: the repair campaign for the toy, r2-lucid for LUCiD data."""
    import os
    if os.environ.get("R2_REPAIR_OUTPUT_ROOT"):
        return Path(os.environ["R2_REPAIR_OUTPUT_ROOT"])
    return ROOT if dr.check(kind) == dr.ANALYTIC_TOY else RETURNS / LUCID_EXTENSION / "train"
from .lucid_views import ViewMasker
from .point_encoder import EncoderConfig
from .r2_probe_repair import fit_ridge, moment_features


class EncoderReadout(torch.nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm

    def encode(self, features, hidden_mask=None):
        return self.arm.encode_tokens(features, hidden_mask=hidden_mask)


def cohort_sample(cohort):
    photon = []
    for i in range(cohort.n_runs):
        truth = cohort.run_truth(i)
        photon.append(truth["event_ids"][truth["domain"].astype(str) == lb.DOMAIN_PHOTON])
    rng = lb.keyed_rng(SEED, "r2-repair/evolution", cohort.name)
    return np.sort(rng.choice(np.concatenate(photon), 1000, replace=False))


@torch.inference_mode()
def evolution_stream(arm, geometry, scaling, cohort, chosen, device, chunk_size, *, layout):
    masker = ViewMasker(geometry, "within", seed=0, n_min=0)
    rows = {name: [] for name in ("all_mean", "spatial", "vertex", "log_yield",
                                 "run_ids", "event_ids")}
    state = er.layout_state(geometry.n_pmt, arm.d_model) if layout else None
    for i, run in enumerate(cohort.run_ids):
        truth = cohort.run_truth(i)
        selected = np.flatnonzero(np.isin(truth["event_ids"], chosen))
        if not selected.size:
            continue
        store = cohort.run_store(i)
        for first in range(0, selected.size, chunk_size):
            indices = selected[first:first + chunk_size]
            ids = truth["event_ids"][indices]
            batch = rs.tokenise_sparse(store, geometry, event_ids=ids, scaling=scaling, masker=masker)
            if not np.array_equal(batch.event_ids, ids):
                raise ValueError(f"snapshot sample lost alignment in {run}")
            tokens = arm.encode_tokens(batch.features.to(device)).float().cpu().numpy()
            if state is not None:
                er.layout_add(state, tokens)
            spatial, _ = moment_features(tokens, batch.features.numpy())
            rows["spatial"].append(spatial)
            rows["all_mean"].append(tokens.mean(axis=1))
            rows["vertex"].append(truth["vertex_m"][indices])
            rows["log_yield"].append(np.log(truth["yield_photons"][indices]))
            rows["run_ids"].append(np.repeat(run, ids.size))
            rows["event_ids"].append(ids)
    rows = {name: np.concatenate(parts) for name, parts in rows.items()}
    if not np.array_equal(np.sort(rows["event_ids"]), chosen):
        raise ValueError("snapshot sample does not match the selected event IDs")
    return rows, er.layout_fractions(state) if state is not None else None


def checkpoint(arm, directory, step, record):
    path = directory / f"snapshot_step_{step:05d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    checks = (payload["step"], payload["identity"], payload["config_sha256"])
    expected = (step, record["recipe"]["identity"], record["config_sha256"])
    if checks != expected:
        raise ValueError(f"snapshot identity mismatch at {path}")
    arm.load_state_dict(payload["model"])
    return {"file": str(path), "sha256": rs.sha256_file(path), "step": step,
            "training_metrics": payload["metrics"]}


def time_share(metrics):
    if not {"time", "hit", "charge"} <= metrics.keys():
        return None
    total = sum(metrics[key] for key in ("time", "hit", "charge"))
    return metrics["time"] / total if total > 0 else None


def evaluate(args):
    kind = dr.check(getattr(args, "data_kind", None))
    seed = int(getattr(args, "seed", SEED) or SEED)
    root = campaign_root(kind)
    from .ext_r2_repair_train import ts_suffix
    tag = run_id(args.detector, args.arm, seed) + ts_suffix(args)
    directory = args.run_dir or root / "runs" / tag
    output = root / "retrain" / "by_run" / tag
    output.mkdir(parents=True, exist_ok=True)
    geometry, scaling = geometry_and_scaling(args.detector, kind)
    h5 = args.data or probes.dataset_path(args.detector, kind)
    cohorts = {split: rs.open_cohort(h5, f"dev_probe_{split}") for split in ("train", "test")}
    if np.intersect1d(cohorts["train"].run_ids, cohorts["test"].run_ids).size:
        raise ValueError("overlapping probe train and test runs")
    if args.arm == "init":  # random initialisation: the final suites only (no training record)
        if (output / "final.json").is_file():
            return
        arm = r2.build_r2_arm(r2.INIT_ARM, geometry, seed, scaling=scaling).to(args.device).eval()
        return _final(args, arm, geometry, scaling, cohorts, output, tag, seed, None,
                      {"checkpoint": None, "step": 0}, kind)
    record = json.loads((directory / "run_record.json").read_text())
    if not record["complete"] or record["step"] != r2.STEPS:
        raise ValueError(f"{tag}: evaluation requires the complete 10,000-step run")
    encoder = EncoderConfig(**record["recipe"]["encoder"])
    arm = r2.build_r2_arm(args.arm, geometry, seed, scaling=scaling,
                          encoder_config=encoder).to(args.device).eval()
    if args.detector in ("wcte", "sk") and args.include != "final":
        samples = {split: cohort_sample(cohort) for split, cohort in cohorts.items()}
        sequence = []
        for step in SNAPSHOTS:
            cached = output / f"step_{step:05d}.json"
            meta = checkpoint(arm, directory, step, record)
            if cached.is_file():
                row = json.loads(cached.read_text())
                if row["checkpoint"] != meta or not cached.with_suffix(".npz").is_file():
                    raise ValueError(f"stale snapshot evaluation at {cached}")
                sequence.append(row)
                continue
            train, _ = evolution_stream(arm, geometry, scaling, cohorts["train"], samples["train"],
                                        args.device, args.chunk, layout=False)
            test, layout = evolution_stream(arm, geometry, scaling, cohorts["test"], samples["test"],
                                            args.device, args.chunk, layout=True)
            vertex = {}
            for form in ("all_mean", "spatial"):
                prediction, alpha = fit_ridge(train[form], train["vertex"], train["run_ids"],
                                              test[form], form=form, target_names=probes.XYZ)
                key = "spatial_moment_latent" if form == "spatial" else form
                vertex[key] = dict(probes.score_vertex(prediction, test["vertex"],
                                                       probes.R_FID[args.detector]), alpha=alpha)
            row = {"step": step, "checkpoint": meta, "layout": layout, "vertex": vertex,
                   "effective_rank": er.effective_rank(test["all_mean"])["effective_rank"],
                   "mae_time_loss_share": time_share(meta["training_metrics"]),
                   "n_train": train["event_ids"].size, "n_test": test["event_ids"].size}
            np.savez_compressed(cached.with_suffix(".npz"), **test, train_event_ids=train["event_ids"])
            write_json(cached, row)
            sequence.append(row)
            print(tag, step, "snapshot evaluated", flush=True)
        write_json(output / "evolution.json", {"run_id": tag, "snapshots": sequence})
    if args.include != "evolution":
        meta = checkpoint(arm, directory, r2.STEPS, record)
        if (output / "final.json").is_file():
            final = json.loads((output / "final.json").read_text())
            if final["checkpoint"] != meta:
                raise ValueError(f"{tag}: stale final evaluation")
            return
        _final(args, arm, geometry, scaling, cohorts, output, tag, seed, record, meta, kind,
               directory=directory)


def _final(args, arm, geometry, scaling, cohorts, output, tag, seed, record, meta, kind,
           directory=None):
    streamed = r2.stream_forms(EncoderReadout(arm), geometry, scaling, cohorts,
                               device=args.device, chunk_events=args.chunk)
    standard, standard_arrays = r2.metrics_from_streams(streamed, geometry)
    del streamed
    repaired, arrays = probes.evaluate(arm, args.detector, seed, geometry, scaling,
                                       list(cohorts.values()), device=args.device,
                                       chunk_size=args.chunk)
    if not np.array_equal(standard_arrays["event_ids"], arrays["event_ids"]):
        raise ValueError(f"{tag}: final probe sets lost alignment")
    mae = args.arm in ("mae_mse", "mae_repaired")
    if mae and directory is not None:
        lines = (json.loads(line) for line in (directory / "train_log.jsonl").read_text().splitlines())
        by_step = {line["step"]: line for line in lines}
        if set(by_step) != set(range(1, r2.STEPS + 1)):
            raise ValueError(f"{tag}: MAE loss log is missing training steps")
        standard["mae_loss_shares"] = r2.loss_shares(by_step.values())
    np.savez_compressed(output / "predictions.npz", **arrays,
                        standard_vertex_errors=standard_arrays["vertex_errors"])
    out = {"run_id": tag, "detector": args.detector, "arm": args.arm, "seed": seed,
           "data_kind": kind, "training": record, "checkpoint": meta, "standard": standard,
           "repaired": repaired}
    write_json(output / "final.json", out)
    print(tag, "final evaluated", flush=True)


def aggregate(args):
    final, evolution = {}, {}
    for detector, arms in MATRIX.items():
        for arm in arms:
            tag = run_id(detector, arm)
            directory = ROOT / "retrain" / "by_run" / tag
            final[tag] = json.loads((directory / "final.json").read_text())
            if detector in ("wcte", "sk"):
                series = json.loads((directory / "evolution.json").read_text())["snapshots"]
                if tuple(row["step"] for row in series) != SNAPSHOTS:
                    raise ValueError(f"{tag}: incomplete evolution series")
                evolution[tag] = series
    args.output.mkdir(parents=True, exist_ok=True)
    info = {"seed": SEED, "registered": False}
    write_json(args.output / "retrain_readings.json", dict(info, runs=final))
    write_json(args.output / "latent_evolution.json",
               dict(info, snapshots=SNAPSHOTS, events_per_probe_cohort=1000, runs=evolution))



def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .ext_r2_repair_figs import RANDOM_STATE, _project

    detector = args.detector
    arms = MATRIX[detector]
    sources = {arm: ROOT / "retrain" / "by_run" / run_id(detector, arm) for arm in arms}
    series = {arm: json.loads((path / "evolution.json").read_text())["snapshots"]
              for arm, path in sources.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(19, 9), layout="constrained")
    fields = (
        ("Event variance fraction", lambda r: r["layout"]["f_event"]),
        ("Layout variance fraction", lambda r: r["layout"]["f_layout"]),
        ("Event-mean effective rank", lambda r: r["effective_rank"]),
        ("All-mean Ridge vertex error (m)", lambda r: r["vertex"]["all_mean"]["mean_m"]),
        ("Spatial-moment Ridge vertex error (m)",
         lambda r: r["vertex"]["spatial_moment_latent"]["mean_m"]),
        ("Training prediction loss", lambda r: r["checkpoint"]["training_metrics"].get(
            "prediction", r["checkpoint"]["training_metrics"].get("loss"))),
        ("Training SIGReg loss", lambda r: r["checkpoint"]["training_metrics"].get("sigreg")),
        ("MAE time loss share", lambda r: r["mae_time_loss_share"]),
    )
    for ax, (label, getter) in zip(axes.flat, fields):
        for arm, rows in series.items():
            if tuple(row["step"] for row in rows) != SNAPSHOTS:
                raise ValueError(f"{detector}/{arm}: incomplete evolution plot inputs")
            y = np.asarray([getter(row) for row in rows], dtype=float)
            ax.plot(range(len(SNAPSHOTS)), y, marker="o", label=arm)
        ax.set(xlabel="Optimizer step", ylabel=label, xticks=range(len(SNAPSHOTS)),
               xticklabels=[str(step) for step in SNAPSHOTS])
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f"{detector.upper()} training evolution across 1,000 held-out photon events per snapshot")
    figure.savefig(args.output / f"metrics_vs_step_{detector}.png", dpi=145)
    plt.close(figure)

    figure, axes = plt.subplots(2 * len(arms), len(SNAPSHOTS), figsize=(22, 3.25 * len(arms)),
                                layout="constrained", squeeze=False)
    heldout = None
    maps = {}
    for i, (arm, path) in enumerate(sources.items()):
        maps[arm] = {}
        for j, step in enumerate(SNAPSHOTS):
            with np.load(path / f"step_{step:05d}.npz", allow_pickle=False) as data:
                values = data["all_mean"]
                colors = (np.linalg.norm(data["vertex"], axis=1), data["log_yield"])
                ids = data["event_ids"]
            if heldout is not None and not np.array_equal(heldout, ids):
                raise ValueError("plot event order differs across checkpoints")
            heldout = ids
            embedding = _project(values, "umap")
            maps[arm][str(step)] = embedding.astype(float).tolist()
            for k, (color, name, cmap) in enumerate(zip(colors, ("Vertex radius (m)", "Log yield"),
                                                       ("viridis", "plasma"))):
                ax = axes[2 * i + k, j]
                drawn = ax.scatter(*embedding.T, c=color, cmap=cmap, s=2, linewidths=0,
                                   vmin=color.min(), vmax=color.max())
                ax.set(xticks=[], yticks=[])
                if i == 0 and k == 0:
                    ax.set_title(str(step))
                if j == 0:
                    ax.set_ylabel(f"{arm}\n{name}", fontsize=8)
                if j == len(SNAPSHOTS) - 1:
                    figure.colorbar(drawn, ax=ax, shrink=0.65)
    figure.suptitle(f"{detector.upper()} UMAP of identical held-out events at each training step")
    figure.savefig(args.output / f"umap_evolution_{detector}.png", dpi=135)
    plt.close(figure)
    write_json(ROOT / "retrain" / f"umap_coordinates_{detector}.json",
               {"event_ids": heldout.tolist(), "cohort_seed": SEED,
                "projection_seed": RANDOM_STATE, "coordinates": maps})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("evaluate")
    one.add_argument("detector", choices=MATRIX)
    one.add_argument("arm", choices=sorted(set().union(*MATRIX.values(), *LUCID_MATRIX.values(),
                                                       {"init"})))
    one.add_argument("--seed", type=int, default=SEED)
    one.add_argument("--data-kind", choices=dr.DATA_KINDS, default=dr.DEFAULT_DATA_KIND)
    one.add_argument("--data", type=Path)
    one.add_argument("--time-loss-scale-ns", type=float, default=None)
    one.add_argument("--run-dir", type=Path)
    one.add_argument("--device", default="cuda")
    one.add_argument("--chunk", type=int, default=8)
    one.add_argument("--include", choices=("all", "final", "evolution"), default="all")
    combine = sub.add_parser("aggregate")
    combine.add_argument("--output", type=Path, required=True)
    figures = sub.add_parser("plot")
    figures.add_argument("detector", choices=("wcte", "sk"))
    figures.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"evaluate": evaluate, "aggregate": aggregate, "plot": plot}[args.command](args)


if __name__ == "__main__":
    main()
