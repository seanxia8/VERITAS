#!/usr/bin/env python3
"""Unregistered de-jitter extension driver (runbook 2026-10-30, reviewed).

Modes
-----
Part 1 (CPU, Deepthought)
  sens-gen       1 000-event five-input counterfactual feature sets (§2.2)
  sens-encode    one frozen encoder on the five inputs
  sens-aggregate sensitivities, ratios and whitened ratios, plus raw rows
  detect-gen     one shard of the 10 000-event R = 8 trigger dataset (§2.3)
  detect-merge   concatenate the shards in run order
  detect-encode  one frozen encoder on the 10 000 r = 0 events
  detect-aggregate  robust/mean delta and vertex probes, raw ceiling, hit-time row
Part 2
  linear         §3.2 linear check with and without median_time
  train          one of the six approved L40S runs (mode_train copied)
  inspect        Part C inspect path (mode_inspect copied), used for the floors
  report         results/jepa/20261030_dejitter/

Everything is an extension (``registered: false``).  No registered code or
config is edited; the median_time wrapper is a runtime patch inside this
process only.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from jepa_noise import ext_dejitter as ed
from jepa_noise import lucid_bombs as lb

REPO = Path(__file__).resolve().parents[1]
EXT = Path("/ceph/dwong/jepa/returns/ext_dejitter_20261030")
M_DOSE_EXT = Path("/ceph/dwong/jepa/returns/ext_m_dose_20261030")
M_DOSE_RESULTS = REPO / "results/jepa/20261030_m_dose"
TIMING_RESULTS = REPO / "results/jepa/20261030_timing"
RESULTS = REPO / "results/jepa/20261030_dejitter"
CONFIG = "src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json"
GEN_CONFIG = "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json"
ARMS_CONFIG = "src/jepa_noise/configs/lucid_arms_v0.json"
GEOMETRY = "/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json"
DATA_ROOT = "/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/"
SEED = 20261017
SPREAD = 0.05
N_RUNS = 100
N_EVENTS = 10
R_REALISATIONS = 8
N_SENS_EVENTS = N_RUNS * N_EVENTS
N_DETECT_EVENTS = 1000 * 10
INPUTS = ("x", "jitp", "jitm", "noise", "phys")
TRAIN_VARIANTS = ("across_median", "within_median", "within_trigger")
EXTENSION = "dejitter_2026-10-30"


# ----------------------------------------------------------------- utilities
def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path, value, overwrite=False):
    raw = (json.dumps(jsonable(value), indent=1, allow_nan=False) + "\n").encode()
    if len(raw) > 1_000_000:
        raise ValueError(f"{path}: summary exceeds 1 MB")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb" if overwrite else "xb") as stream:
        stream.write(raw)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_storage():
    root = EXT.resolve()
    if not root.is_relative_to(Path("/ceph/dwong/jepa")):
        raise SystemExit(f"extension root {root} is outside /ceph/dwong/jepa")
    free = None
    try:
        free = (int(os.getxattr("/ceph/dwong", "ceph.quota.max_files"))
                - int(os.getxattr("/ceph/dwong", "ceph.dir.rfiles"))
                - int(os.getxattr("/ceph/dwong", "ceph.dir.rsubdirs")))
    except OSError:
        pass
    entries = size = 0
    if root.exists():
        for base, dirs, files in os.walk(root):
            entries += len(dirs) + len(files)
            for name in files:
                try:
                    size += (Path(base) / name).stat().st_size
                except FileNotFoundError:
                    continue
    if free is not None and free < 30000:
        raise SystemExit(f"S1: {free} free entries < 30000")
    if entries >= 3000 or size >= 5_000_000_000:
        raise SystemExit(f"S1: extension storage {entries} entries / {size} bytes "
                         f"exceeds the 3000-entry / 5 GB budget")
    return {"free_entries": free, "extension_entries": entries,
            "extension_bytes": size}


def check_provenance():
    run_sha = sha256_file(CONFIG)
    gen_sha = sha256_file(GEN_CONFIG)
    if not run_sha.startswith("e6730ac1"):
        raise SystemExit(f"run config sha256 {run_sha} does not match e6730ac1")
    if not gen_sha.startswith("90ac1601"):
        raise SystemExit(f"generator config sha256 {gen_sha} does not match 90ac1601")
    geometry = lb.load_wcte_geometry(GEOMETRY)
    return {"run_config": {"path": CONFIG, "sha256": run_sha},
            "generator_config": {"path": GEN_CONFIG, "sha256": gen_sha},
            "geometry": {"path": GEOMETRY, "sha256": sha256_file(GEOMETRY),
                         "digest": geometry.digest},
            "storage_before": check_storage()}


def setup(args):
    from jepa_noise import lucid_arms as la
    from jepa_noise import lucid_run as lr
    config = lr.load_run_config(args.config)
    gen = lb.load_generator_config(args.generator_config)
    arms = la.load_config(args.arms_config)
    geometry = lb.load_wcte_geometry(args.geometry)
    device = lr.configure_runtime(args.device)
    return config, gen, arms, geometry, device


def cells_for(config):
    from scripts import jepa_pregrid_checks as pregrid
    return {
        "dev": pregrid.lr.dev_cell(config),
        "across": pregrid.variant_cell(config, "across"),
        "across_median": dataclasses.replace(
            pregrid.variant_cell(config, "across"),
            cell_id="ext_dev_s005_across_median"),
        "within_median": pregrid.variant_cell(config, "median"),
        "within_trigger": pregrid.variant_cell(config, "trigger"),
    }


def encoder_specs():
    from scripts.jepa_ext_timing import _checkpoint_specs
    return _checkpoint_specs()


def build_model(spec, config, gen, arms, geometry, device="cpu"):
    from scripts import jepa_pregrid_checks as pregrid
    cells = cells_for(config)
    cell = cells["across"] if spec["cell"] == "across" else cells["dev"]
    model, beta = pregrid.build_loaded(
        spec["arm"], cell, geometry, arms, config, gen, Path(DATA_ROOT),
        spec["source"], spec["seed"])
    return getattr(model, "jepa", model).to(device).eval()


def encode_features(model, features, chunk=64):
    import torch
    vectors = []
    with torch.no_grad():
        for start in range(0, len(features), chunk):
            batch = torch.as_tensor(np.asarray(features[start:start + chunk]),
                                    dtype=torch.float32)
            tokens = model.encode(batch)
            vectors.append(tokens.mean(dim=1).cpu().numpy())
    return np.concatenate(vectors).astype(np.float32)


def _tools(geometry, gen):
    from jepa_noise import point_encoder as pe
    from jepa_noise.lucid_views import ViewMasker
    from scripts.jepa_ext_m_dose import view_groups
    return {"readout": lb.make_readout(geometry, gen),
            "scaling": pe.FeatureScaling.from_geometry(geometry),
            "masker": ViewMasker(geometry, "within", n_min=0),
            "groups": view_groups(geometry)}


def _tokenise(geometry, hits, tools):
    from scripts.jepa_ext_m_dose import tokenise_hits
    return tokenise_hits(geometry, hits, tools["scaling"], tools["masker"])


def _pooled(features, split, tools):
    from scripts.jepa_ext_m_dose import pooled
    return pooled(features, split, tools["groups"])


def _state(geometry, law, spread, run_id, seed=SEED):
    from scripts.jepa_ext_m_dose import state_for
    return state_for(geometry, law, spread, run_id, seed=seed)


# ------------------------------------------------------- Part 1 §2.2 inputs
def mode_sens_gen(args):
    from jepa_noise import ext_timing as et
    from jepa_noise.ext_predictability import resample_noise
    config, gen, arms, geometry, device = setup(args)
    tools = _tools(geometry, gen)
    real_rule = lb.pmt_time_rule()
    original_rule = lb._PMT_TIME_RULE
    folder = EXT / "part1" / "sens"
    folder.mkdir(parents=True, exist_ok=True)
    features = {name: np.lib.format.open_memmap(
        folder / f"features_{name}.npy", mode="w+", dtype=np.float32,
        shape=(N_SENS_EVENTS, geometry.n_pmt, 9)) for name in INPUTS}
    dark = np.zeros((N_SENS_EVENTS, len(INPUTS)), dtype=bool)
    vertex = np.empty((N_SENS_EVENTS, 3))
    vertex_phys = np.empty((N_SENS_EVENTS, 3))
    emission = np.empty(N_SENS_EVENTS)
    triggers = np.full((N_SENS_EVENTS, len(INPUTS)), np.nan)
    time_sd = np.empty(N_SENS_EVENTS)
    redraws = np.zeros((N_SENS_EVENTS, 2), dtype=np.int64)
    cursor = 0
    try:
        for run in range(N_RUNS):
            run_id = f"ext_m_dose/evt/{run:04d}"
            state = _state(geometry, "F1", SPREAD, run_id)
            hits = {name: [] for name in INPUTS}
            for event in range(N_EVENTS):
                event_id = run * N_EVENTS + event
                draws = lb.draw_event(geometry, gen, run_id, event_id,
                                      seed=SEED, domain="photon")
                lb._PMT_TIME_RULE = real_rule
                records = {"x": tools["readout"](state, draws)}
                lb._PMT_TIME_RULE = et.shifted_time_rule(real_rule, 1.0)
                records["jitp"] = tools["readout"](state, draws)
                lb._PMT_TIME_RULE = et.shifted_time_rule(real_rule, -1.0)
                records["jitm"] = tools["readout"](state, draws)
                lb._PMT_TIME_RULE = real_rule
                sample = resample_noise(draws, geometry, gen, 1, seed=SEED)
                records["noise"] = tools["readout"](state, sample)
                displaced, info = ed.sample_physical_vertex(
                    draws, geometry, gen, seed=SEED, run_id=run_id,
                    event_id=event_id)
                records["phys"] = tools["readout"](state, displaced)
                change = ed.propagation_time_change(
                    geometry, gen, draws.vertex_m, displaced.vertex_m)
                index = cursor + event
                dark[index] = [ed.dark_moved(records[name]) for name in INPUTS]
                vertex[index] = draws.vertex_m
                vertex_phys[index] = displaced.vertex_m
                emission[index] = draws.emission_offset_ns
                time_sd[index] = float(np.std(change))
                redraws[index] = (info["fiducial_redraws"], info["capacity_failures"])
                for name in INPUTS:
                    record = records[name]
                    triggers[index, INPUTS.index(name)] = (
                        np.nan if record.trigger_ns is None else float(record.trigger_ns))
                    hits[name].append(record)
            for name in INPUTS:
                batch, _ = _tokenise(geometry, hits[name], tools)
                features[name][cursor:cursor + N_EVENTS] = batch.astype(np.float32)
            cursor += N_EVENTS
            if run % 10 == 0:
                print(f"sens-gen: run {run}/{N_RUNS}", flush=True)
    finally:
        lb._PMT_TIME_RULE = original_rule
    for name in INPUTS:
        features[name].flush()
    np.savez_compressed(
        folder / "meta.npz", dark=dark, vertex=vertex, vertex_phys=vertex_phys,
        emission=emission, triggers=triggers, time_change_sd=time_sd,
        redraws=redraws,
        valid=~dark.any(axis=1))
    write_json(folder / "generation.json", {
        "registered": False, "extension": EXTENSION, "kind": "sens_gen",
        "provenance": check_provenance(), "n_events": N_SENS_EVENTS,
        "inputs": list(INPUTS),
        "excluded_dark_moved": int(dark.any(axis=1).sum()),
        "fiducial_redraws": int(redraws[:, 0].sum()),
        "capacity_failures": int(redraws[:, 1].sum()),
        "time_change_sd_mean_ns": float(np.mean(time_sd)),
        "time_change_sd_of_event_sd_ns": float(np.std(time_sd)),
        "invocation": "PYTHONPATH=src python -m scripts.jepa_ext_dejitter sens-gen",
    })
    print("SENS GEN OK", flush=True)


def mode_sens_encode(args):
    config, gen, arms, geometry, device = setup(args)
    spec = encoder_specs()[args.label]
    model = build_model(spec, config, gen, arms, geometry, "cpu")
    folder = EXT / "part1" / "sens"
    output = {}
    for name in INPUTS:
        features = np.load(folder / f"features_{name}.npy", mmap_mode="r")
        output[name] = encode_features(model, features)
        del features
    np.savez_compressed(folder / f"z_{args.label}.npz", **output)
    write_json(folder / f"z_{args.label}.json", {
        "registered": False, "extension": EXTENSION, "kind": "sens_encode",
        "label": args.label, "checkpoint_sha256": spec.get("checkpoint_sha256"),
        "inputs": list(INPUTS), "n_events": N_SENS_EVENTS,
        "invocation": "PYTHONPATH=src python -m scripts.jepa_ext_dejitter sens-encode",
    })
    print(f"SENS ENCODE OK {args.label}", flush=True)


def _ratio(numerator, denominator):
    return None if denominator in (None, 0.0) else float(numerator / denominator)


def mode_sens_aggregate(args):
    folder = EXT / "part1" / "sens"
    meta = np.load(folder / "meta.npz")
    config, gen, arms, geometry, device = setup(args)
    tools = _tools(geometry, gen)
    valid = meta["valid"]
    rows = {}
    labels = list(encoder_specs())
    for label in labels:
        with np.load(folder / f"z_{label}.npz") as data:
            z = {name: data[name].astype(np.float64) for name in INPUTS}
        deltas = {name: z[name] - z["x"] for name in INPUTS if name != "x"}
        deltas = {name: value[valid] for name, value in deltas.items()}
        base = ed.sensitivity_terms(z["x"][valid], deltas["phys"])
        sensed = {name: ed.sensitivity_terms(z["x"][valid], deltas[name])["mean_delta_squared"]
                  for name in deltas}
        whitened = ed.whitened_sensitivity_terms(
            z["x"][valid], deltas, threshold=1e-6)
        jit = 0.5 * (sensed["jitp"] + sensed["jitm"])
        jit_w = 0.5 * (whitened["jitp"] + whitened["jitm"])
        rows[label] = {
            "n_events": int(valid.sum()),
            "sensitivity_jit": _ratio(jit, base["trace_covariance"]),
            "sensitivity_noise": _ratio(sensed["noise"], base["trace_covariance"]),
            "sensitivity_phys": _ratio(sensed["phys"], base["trace_covariance"]),
            "R_jit_phys": _ratio(jit, sensed["phys"]),
            "R_jit_noise": _ratio(jit, sensed["noise"]),
            "whitened_rank": whitened["rank"],
            "whitened_sensitivity_jit": _ratio(jit_w, whitened["trace_covariance"]),
            "whitened_R_jit_phys": _ratio(jit_w, whitened["phys"]),
            "whitened_R_jit_noise": _ratio(jit_w, whitened["noise"]),
        }
    raw = {}
    features = {name: np.load(folder / f"features_{name}.npy", mmap_mode="r")
                for name in INPUTS}
    for name in INPUTS:
        raw[name] = np.empty((N_SENS_EVENTS, 315 * 2))
    for name in INPUTS:
        for start in range(0, N_SENS_EVENTS, 200):
            raw[name][start:start + 200] = _pooled(
                np.asarray(features[name][start:start + 200], dtype=np.float64),
                "within", tools)
        del features[name]
    base = ed.sensitivity_terms(raw["x"][valid], raw["phys"][valid] - raw["x"][valid])
    sensed = {name: ed.sensitivity_terms(
        raw["x"][valid], raw[name][valid] - raw["x"][valid])["mean_delta_squared"]
        for name in INPUTS if name != "x"}
    whitened = ed.whitened_sensitivity_terms(
        raw["x"][valid],
        {name: raw[name][valid] - raw["x"][valid] for name in INPUTS if name != "x"},
        threshold=1e-6)
    jit = 0.5 * (sensed["jitp"] + sensed["jitm"])
    raw_row = {
        "n_events": int(valid.sum()),
        "sensitivity_jit": _ratio(jit, base["trace_covariance"]),
        "sensitivity_noise": _ratio(sensed["noise"], base["trace_covariance"]),
        "sensitivity_phys": _ratio(sensed["phys"], base["trace_covariance"]),
        "R_jit_phys": _ratio(jit, sensed["phys"]),
        "R_jit_noise": _ratio(jit, sensed["noise"]),
        "whitened_rank": whitened["rank"],
        "whitened_R_jit_phys": _ratio(0.5 * (whitened["jitp"] + whitened["jitm"]),
                                      whitened["phys"]),
    }
    write_json(EXT / "part1" / "sensitivity.json", {
        "registered": False, "extension": EXTENSION, "kind": "part1_sensitivity",
        "events": N_SENS_EVENTS,
        "excluded_dark_moved": int((~valid).sum()),
        "redraw_counts": {"fiducial": int(meta["redraws"][:, 0].sum()),
                          "capacity": int(meta["redraws"][:, 1].sum())},
        "time_change_sd_ns": float(np.mean(meta["time_change_sd"])),
        "encoders": rows,
        "raw_rows": raw_row,
    })
    print("SENS AGGREGATE OK", flush=True)


# ------------------------------------------------- Part 1 §2.3 10k events
def mode_detect_gen(args):
    from jepa_noise.ext_predictability import resample_noise
    config, gen, arms, geometry, device = setup(args)
    tools = _tools(geometry, gen)
    folder = EXT / "part1" / "detect"
    folder.mkdir(parents=True, exist_ok=True)
    runs_per_shard = 1000 // args.shards
    start_run = args.shard * runs_per_shard
    stop_run = start_run + runs_per_shard
    n_events = runs_per_shard * N_EVENTS
    features = np.lib.format.open_memmap(
        folder / f"features_{args.shard:02d}.npy", mode="w+", dtype=np.float32,
        shape=(n_events, geometry.n_pmt, 9))
    triggers = np.full((n_events, R_REALISATIONS), np.nan)
    dark = np.zeros((n_events, R_REALISATIONS), dtype=bool)
    vertex = np.empty((n_events, 3))
    emission = np.empty(n_events)
    cursor = 0
    for run in range(start_run, stop_run):
        run_id = f"ext_m_dose/evt/{run:04d}"
        state = _state(geometry, "F1", SPREAD, run_id)
        hits = []
        for event in range(N_EVENTS):
            draws = lb.draw_event(geometry, gen, run_id, run * N_EVENTS + event,
                                  seed=SEED, domain="photon")
            records = [tools["readout"](state, resample_noise(
                draws, geometry, gen, replicate, seed=SEED))
                for replicate in range(R_REALISATIONS)]
            for replicate, record in enumerate(records):
                triggers[cursor + event, replicate] = (
                    np.nan if record.trigger_ns is None else float(record.trigger_ns))
                dark[cursor + event, replicate] = ed.dark_moved(record)
            hits.append(records[0])
            vertex[cursor + event] = draws.vertex_m
            emission[cursor + event] = draws.emission_offset_ns
        batch, _ = _tokenise(geometry, hits, tools)
        features[cursor:cursor + N_EVENTS] = batch.astype(np.float32)
        cursor += N_EVENTS
        if run % 20 == 0:
            print(f"detect-gen shard {args.shard}: run {run}/{stop_run}", flush=True)
    features.flush()
    np.savez_compressed(folder / f"meta_{args.shard:02d}.npz",
                        triggers=triggers, dark=dark, vertex=vertex,
                        emission=emission,
                        valid=~dark.any(axis=1), start_run=start_run)
    print(f"DETECT GEN OK shard {args.shard}", flush=True)


def mode_detect_merge(args):
    folder = EXT / "part1" / "detect"
    metas = [np.load(folder / f"meta_{i:02d}.npz") for i in range(args.shards)]
    features = np.lib.format.open_memmap(
        folder / "features.npy", mode="w+", dtype=np.float32,
        shape=(N_DETECT_EVENTS, lb.load_wcte_geometry(GEOMETRY).n_pmt, 9))
    cursor = 0
    for shard in range(args.shards):
        source = np.load(folder / f"features_{shard:02d}.npy", mmap_mode="r")
        features[cursor:cursor + len(source)] = source
        cursor += len(source)
        del source
    features.flush()
    np.savez_compressed(
        folder / "targets.npz",
        triggers=np.concatenate([m["triggers"] for m in metas]),
        dark=np.concatenate([m["dark"] for m in metas]),
        vertex=np.concatenate([m["vertex"] for m in metas]),
        emission=np.concatenate([m["emission"] for m in metas]),
        valid=np.concatenate([m["valid"] for m in metas]))
    for shard in range(args.shards):
        (folder / f"features_{shard:02d}.npy").unlink(missing_ok=True)
        (folder / f"meta_{shard:02d}.npz").unlink(missing_ok=True)
    print("DETECT MERGE OK", flush=True)


def mode_detect_encode(args):
    config, gen, arms, geometry, device = setup(args)
    spec = encoder_specs()[args.label]
    model = build_model(spec, config, gen, arms, geometry, "cpu")
    folder = EXT / "part1" / "detect"
    features = np.load(folder / "features.npy", mmap_mode="r")
    z = encode_features(model, features, chunk=128)
    np.savez_compressed(folder / f"z_{args.label}.npz", z=z)
    write_json(folder / f"z_{args.label}.json", {
        "registered": False, "extension": EXTENSION, "kind": "detect_encode",
        "label": args.label, "checkpoint_sha256": spec.get("checkpoint_sha256"),
        "n_events": N_DETECT_EVENTS, "dimension": int(z.shape[1]),
        "invocation": "PYTHONPATH=src python -m scripts.jepa_ext_dejitter detect-encode",
    })
    print(f"DETECT ENCODE OK {args.label}", flush=True)


def _probe(vectors, targets, train_mask, test_mask):
    from scripts.jepa_pregrid_checks import ridge_r2
    out = {}
    for name in ("delta_robust", "delta_mean"):
        result = ridge_r2(vectors[train_mask], targets[name][train_mask],
                          vectors[test_mask], targets[name][test_mask])
        out[name] = result["r2"]
    scores = [ridge_r2(vectors[train_mask], targets["vertex"][train_mask, coordinate],
                       vectors[test_mask], targets["vertex"][test_mask, coordinate])["r2"]
              for coordinate in range(3)]
    finite = [value for value in scores if value is not None]
    out["vertex"] = float(np.mean(finite)) if finite else None
    return out


def mode_detect_aggregate(args):
    config, gen, arms, geometry, device = setup(args)
    tools = _tools(geometry, gen)
    folder = EXT / "part1" / "detect"
    with np.load(folder / "targets.npz") as data:
        triggers = data["triggers"].astype(np.float64)
        dark = data["dark"]
        vertex = data["vertex"]
        valid = data["valid"]
    robust = triggers[:, 0] - np.median(triggers, axis=1)
    mean_delta = triggers[:, 0] - triggers.mean(axis=1)
    targets = {"delta_robust": robust, "delta_mean": mean_delta, "vertex": vertex}
    run_ids = np.arange(N_DETECT_EVENTS) // N_EVENTS
    probes = {}
    for label in encoder_specs():
        with np.load(folder / f"z_{label}.npz") as data:
            z = data["z"].astype(np.float64)
        train = valid & (run_ids < 500)
        test = valid & (run_ids >= 500)
        probes[label] = _probe(z, targets, train, test)
        del z
    features = np.load(folder / "features.npy", mmap_mode="r")
    pooled_features = np.empty((N_DETECT_EVENTS, 630))
    for start in range(0, N_DETECT_EVENTS, 200):
        pooled_features[start:start + 200] = _pooled(
            np.asarray(features[start:start + 200], dtype=np.float64), "within", tools)
    hit = np.asarray(features[..., ed.HIT_INDEX]) > 0
    time = np.asarray(features[..., ed.TIME_INDEX])
    mean_hit_time = np.where(hit, time, 0.0).sum(axis=1) / np.maximum(hit.sum(axis=1), 1)
    del features
    train = valid & (run_ids < 500)
    test = valid & (run_ids >= 500)
    ceilings = {"raw_pooled": _probe(pooled_features, targets, train, test),
                "hit_token_mean_time": _probe(mean_hit_time[:, None], targets,
                                              train, test)}
    write_json(folder / "detectability.json", {
        "registered": False, "extension": EXTENSION, "kind": "part1_detectability",
        "n_events": N_DETECT_EVENTS,
        "excluded_dark_moved": int((~valid).sum()),
        "target_sd": {"delta_robust": float(np.std(robust[valid])),
                      "delta_mean": float(np.std(mean_delta[valid]))},
        "encoders": probes, "ceilings": ceilings,
        "train": "runs 0-499", "test": "runs 500-999",
    })
    print("DETECT AGGREGATE OK", flush=True)


# ------------------------------------------------------ Part 2 §3.2 linear
def mode_linear(args):
    from jepa_noise.ext_predictability import resample_noise
    from jepa_noise import ext_timing as et
    from scripts.jepa_ext_timing import load_level, _rows
    config, gen, arms, geometry, device = setup(args)
    tools = _tools(geometry, gen)
    scale = np.sqrt(R_REALISATIONS / (R_REALISATIONS - 1))
    output = {"registered": False, "extension": EXTENSION, "kind": "linear_check",
              "provenance": check_provenance(), "levels": {}}
    for law, spread in (("F1", 0.00), ("F1", 0.05)):
        name = f"{law}_{spread:.2f}"
        arrays, _ = load_level(law, spread)
        blocks_store = {}
        for split in ("within", "across_fixed"):
            context_dim = int(arrays[f"{split}_context_dim"])
            blocks_store[split] = (context_dim,
                                   int(arrays[f"{split}_target_dim"]))
        accum = {split: {key: np.zeros_like(arrays[f"{split}_A"])
                         for key in ("A", "B", "Amed", "Bmed")}
                 for split in blocks_store}
        started = time.monotonic()
        for event in range(1000):
            run_id = f"ext_m_dose/cal/{event:04d}"
            draws = lb.draw_event(geometry, gen, run_id, event, seed=SEED,
                                  domain="photon")
            fixed_state = _state(geometry, law, spread, run_id)
            hits_a, hits_b = [], []
            for replicate in range(R_REALISATIONS):
                record_a = tools["readout"](fixed_state, resample_noise(
                    draws, geometry, gen, replicate, seed=SEED))
                hits_a.append(record_a)
                fresh_seed = int(lb.keyed_rng(
                    SEED, "ext_predictability_state", run_id, event, replicate)
                    .integers(0, 2**64, dtype=np.uint64))
                fresh_state = _state(geometry, law, spread, run_id, seed=fresh_seed)
                record_b = tools["readout"](fresh_state, resample_noise(
                    draws, geometry, gen, replicate + 1, seed=SEED))
                hits_b.append(record_b)
            features_a, _ = _tokenise(geometry, hits_a, tools)
            features_b, _ = _tokenise(geometry, hits_b, tools)
            medians_a = ed.median_hit_time(features_a)
            medians_b = ed.median_hit_time(features_b)
            for split, (context_dim, target_dim) in blocks_store.items():
                xa = _pooled(features_a, split, tools)
                xb = _pooled(features_b, split, tools)
                for key, values in (("A", xa), ("B", xb)):
                    residual = (values - values.mean(axis=0)) * scale
                    accum[split][key] += residual.T @ residual / R_REALISATIONS
                # median_time: subtract the per-realisation hit-time median on the
                # pooled time coordinates (t' = t - m * hit fraction).
                group_count = values.shape[1] // 3
                xa_med = xa.copy()
                xb_med = xb.copy()
                for group in range(group_count):
                    xa_med[:, 3 * group + 1] -= (medians_a
                                                 * xa[:, 3 * group + 2])
                    xb_med[:, 3 * group + 1] -= (medians_b
                                                 * xb[:, 3 * group + 2])
                for key, values in (("Amed", xa_med), ("Bmed", xb_med)):
                    residual = (values - values.mean(axis=0)) * scale
                    accum[split][key] += residual.T @ residual / R_REALISATIONS
            if event % 100 == 0:
                print(f"linear {name}: {event}/1000", flush=True)
        level = {"wall_seconds": time.monotonic() - started, "splits": {}}
        for split, (context_dim, target_dim) in blocks_store.items():
            for key in accum[split]:
                accum[split][key] /= 1000
            stored_a = np.asarray(arrays[f"{split}_A"], dtype=np.float64)
            stored_b = np.asarray(arrays[f"{split}_B"], dtype=np.float64)
            diff_a = float(np.max(np.abs(accum[split]["A"] - stored_a)))
            diff_b = float(np.max(np.abs(accum[split]["B"] - stored_b)))
            if diff_a > 1e-9 or diff_b > 1e-9:
                raise SystemExit(f"{name} {split}: regeneration mismatch "
                                 f"{diff_a:.3e}/{diff_b:.3e}")
            S = arrays[f"{split}_S"]
            results = {"regenerated_A_max_abs_difference": diff_a,
                       "regenerated_B_max_abs_difference": diff_b, "variants": {}}
            for variant, cross_a, cross_b in (
                    ("baseline", accum[split]["A"], accum[split]["B"]),
                    ("median_time", accum[split]["Amed"], accum[split]["Bmed"])):
                blocks = et.blocks_from_joint(S, cross_a, cross_b,
                                              context_dim, target_dim)
                results["variants"][variant] = et.pair_metrics(
                    blocks, 4, (blocks["act"], blocks["bct"] - blocks["act"]))
            level["splits"][split] = results
        output["levels"][name] = level
    write_json(EXT / "part2_linear.json", output)
    print("LINEAR OK", flush=True)


# --------------------------------------------------- Part 2 training copies
def _install_wrapper(transform):
    if transform != "median_time":
        return None
    from jepa_noise import point_encoder as pe
    return ed.install_median_time_wrapper(pe)


def _restore_wrapper(original):
    if original is None:
        return
    from jepa_noise import point_encoder as pe
    ed.remove_median_time_wrapper(pe, original)


def mode_train(args):
    from jepa_noise import lucid_run as lr
    from scripts import jepa_pregrid_checks as pregrid
    wrapper = _install_wrapper(args.transform)
    try:
        config, gen, arms, geometry, device = setup(args)
        variant = args.variant
        if variant not in TRAIN_VARIANTS:
            raise SystemExit(f"unknown variant {variant}")
        cell = cells_for(config)[variant]
        arm_name = "sigreg_jepa"
        data_root, out_root = Path(args.data_root), Path(args.out)
        train_ds, probes = pregrid.datasets_for(cell, config, gen, geometry, data_root)
        result = lr.run_one(
            arm_name=arm_name, cell=cell, seed=int(args.seed), beta=None,
            config=config, arms_config=arms, geometry=geometry,
            loader_dataset=train_ds, probe_datasets=probes, dark_bank=None,
            out_root=out_root, data_root=data_root,
            config_sha=lr.sha256_file(args.config),
            code_sha=lr.git_head(REPO), status_text=lr.git_status(REPO),
            device=device, steps=args.steps, evaluate=True,
            probe_max_events=args.probe_max_events, physics_method="ridge",
            save_embeddings=False)
        run_dir = Path(result["run_dir"])
        final = run_dir / "final.pt"
        arm, _ = pregrid.build_loaded(arm_name, cell, geometry, arms, config, gen,
                                      data_root, str(final), args.seed)
        forms = pregrid.evaluate_forms(arm, arm_name, cell, geometry, probes, config,
                                       device, run_dir / "ext_eval",
                                       probe_max_events=args.probe_max_events)
        log = [json.loads(line) for line in
               (run_dir / "train_log.jsonl").read_text().splitlines() if line.strip()]
        summary = {"registered": False, "extension": EXTENSION, "variant": variant,
                   "transform": args.transform, "arm": arm_name,
                   "cell": cell.cell_id, "split": cell.split, "spread": cell.setting,
                   "seed": int(args.seed),
                   "final_train_metrics": log[-1] if log else None,
                   "forms": forms,
                   "wall_seconds": result["manifest"].get("wall_seconds"),
                   "checkpoint": str(final),
                   "checkpoint_sha256": sha256_file(final)}
        (run_dir / "ext_summary.json").write_text(
            json.dumps(jsonable(summary), indent=1, default=float))
        print(json.dumps(jsonable(forms), default=float), flush=True)
    finally:
        _restore_wrapper(wrapper)


def _write_report(out, report, *, complete):
    out.mkdir(parents=True, exist_ok=True)
    report["complete"] = bool(complete)
    temporary = out / "inspect.json.tmp"
    temporary.write_text(json.dumps(jsonable(report), indent=1, default=float))
    temporary.replace(out / "inspect.json")


def mode_inspect(args):
    from scripts import jepa_pregrid_checks as pregrid
    wrapper = _install_wrapper(args.transform)
    try:
        config, gen, arms, geometry, device = setup(args)
        cell = cells_for(config)[args.cell]
        _, probes = pregrid.datasets_for(cell, config, gen, geometry,
                                         Path(args.data_root))
        report = {"registered": False, "extension": EXTENSION, "mode": "inspect",
                  "cell": cell.cell_id, "transform": args.transform, "items": {}}
        for item in args.item:
            label, spec = item.split("=", 1)
            arm_name, rest = spec.split(":", 1)
            if rest.startswith("init:"):
                source, seed = "init", int(rest.split(":", 1)[1])
            else:
                source, seed = rest, 0
            arm, beta = pregrid.build_loaded(arm_name, cell, geometry, arms, config,
                                             gen, Path(args.data_root), source, seed)
            entry = pregrid.evaluate_forms(arm, arm_name, cell, geometry, probes,
                                           config, device, Path(args.out) / label)
            report["items"][label] = {"arm": arm_name, "source": source,
                                      "seed": seed, "beta": beta, **entry}
            print(label, json.dumps(jsonable(entry["full_input"]), default=float),
                  flush=True)
            _write_report(Path(args.out), report, complete=False)
        _write_report(Path(args.out), report, complete=True)
    finally:
        _restore_wrapper(wrapper)


# ------------------------------------------------------------------ report
def mode_report(args):
    sensitivity = json.loads((EXT / "part1" / "sensitivity.json").read_text())
    detectability = json.loads((EXT / "part1" / "detect" / "detectability.json").read_text())
    linear = json.loads((EXT / "part2_linear.json").read_text())
    train_root = Path(args.out) if args.out else EXT / "train"
    summaries = {}
    for path in sorted(train_root.glob("runs/*/ext_summary.json")):
        payload = json.loads(path.read_text())
        summaries[(payload["variant"], payload["seed"])] = payload
    floors = {}
    for path in (EXT / "floors").glob("**/inspect.json"):
        floors[path.parent.name] = json.loads(path.read_text())
    RESULTS.mkdir(parents=True, exist_ok=True)
    # sensitivity table
    rows = []
    for label, value in sensitivity["encoders"].items():
        rows.append([label, _f(value["sensitivity_jit"]), _f(value["sensitivity_noise"]),
                     _f(value["sensitivity_phys"]), _f(value["R_jit_phys"]),
                     _f(value["R_jit_noise"]), _f(value["whitened_R_jit_phys"]),
                     _f(value["whitened_R_jit_noise"]), str(value["whitened_rank"])])
    raw = sensitivity["raw_rows"]
    rows.append(["raw_pooled", _f(raw["sensitivity_jit"]), _f(raw["sensitivity_noise"]),
                 _f(raw["sensitivity_phys"]), _f(raw["R_jit_phys"]),
                 _f(raw["R_jit_noise"]), _f(raw["whitened_R_jit_phys"]), "—",
                 str(raw["whitened_rank"])])
    detection_rows = []
    for label, value in detectability["encoders"].items():
        detection_rows.append([label, _f(value["delta_robust"]),
                               _f(value["delta_mean"]), _f(value["vertex"])])
    for label, value in detectability["ceilings"].items():
        detection_rows.append([label, _f(value["delta_robust"]),
                               _f(value["delta_mean"]), _f(value["vertex"])])
    (RESULTS / "sensitivity.md").write_text(
        "# De-jitter extension — Part 1 sensitivity (registered: false)\n\n"
        f"1 000 events, excluded for dark-moved triggers: "
        f"{sensitivity['excluded_dark_moved']}; fiducial redraws "
        f"{sensitivity['redraw_counts']['fiducial']}, capacity failures "
        f"{sensitivity['redraw_counts']['capacity']}; mean per-PMT x_phys time change SD "
        f"{sensitivity['time_change_sd_ns']:.3f} ns.\n\n"
        "## Counterfactual sensitivities and ratios\n\n"
        + _table(["encoder", "S_jit", "S_noise", "S_phys", "R_jit/phys",
                  "R_jit/noise", "Rw_jit/phys", "Rw_jit/noise", "rank"], rows)
        + "\n\n## 10 000-event detectability probes (robust/mean delta, vertex)\n\n"
        + _table(["row", "R2(delta_robust)", "R2(delta_mean)", "R2(vertex)"],
                 detection_rows) + "\n")
    # prescription table
    prescription = []
    for name, level in linear["levels"].items():
        for split, value in level["splits"].items():
            base = value["variants"]["baseline"]
            med = value["variants"]["median_time"]
            prescription.append([name, split, _f(base["nu_star_A"]),
                                 _f(med["nu_star_A"]), _f(med["nu_star_B"]),
                                 _f(med["nu_star_gain"]),
                                 f"{value['regenerated_A_max_abs_difference']:.1e}/"
                                 f"{value['regenerated_B_max_abs_difference']:.1e}"])
    run_rows = []
    references = {"across": (0.348113310588429, 0.333463657),
                  "within": (0.21925822392091457, 0.2042813949606102)}
    for (variant, seed), payload in sorted(summaries.items()):
        full = payload["forms"]["full_input"]
        rank = (full.get("event_embedding") or {}).get("effective_rank")
        charge = (full.get("token_log_charge_r2") or {}).get("r2")
        run_rows.append([variant, str(seed), _f(full.get("vertex_resolution_m")),
                         _f(full.get("module_gain_run_mean_r2")),
                         _f(full.get("log_yield_r2")), _f(rank),
                         _f(full.get("sigreg_statistic_events")), _f(charge)])
    floor_rows = []
    for name, payload in floors.items():
        item = payload["items"].get("init9000")
        if item:
            full = item["full_input"]
            rank = (full.get("event_embedding") or {}).get("effective_rank")
            charge = (full.get("token_log_charge_r2") or {}).get("r2")
            floor_rows.append([name, _f(full.get("vertex_resolution_m")),
                               _f(full.get("module_gain_run_mean_r2")),
                               _f(rank), _f(charge)])
    readings = _readings(sensitivity, detectability, linear, summaries,
                         floors)
    (RESULTS / "prescription.md").write_text(
        "# De-jitter extension — Part 2 prescription (registered: false)\n\n"
        "## Linear check (§3.2): nu* by split and level, baseline vs median_time\n\n"
        + _table(["level", "split", "nu*(A) baseline", "nu*(A) median_time",
                  "nu*(B) median_time", "nu*(gain) median_time",
                  "regeneration A/B max diff"], prescription)
        + "\n\n## Six new runs (§3.3) vs references\n\n"
        + _table(["variant", "seed", "vertex (m)", "gain R2", "log yield R2",
                  "effective rank", "event SIGReg", "token charge R2"], run_rows)
        + "\n\nPart D across references: 0.348113 / 0.333464 m; Part C fixed-within "
          "references: 0.219258 / 0.204281 m.\n\n"
        "## median_time random-init floors (Titan, inspect with wrapper)\n\n"
        + _table(["cell", "vertex (m)", "gain R2", "effective rank",
                  "token charge R2"], floor_rows) + "\n")
    write_json(RESULTS / "readings.json", readings, overwrite=True)
    write_json(RESULTS / "summary.json",
               {"registered": False, "extension": EXTENSION,
                "provenance": linear["provenance"],
                "part1": {"sensitivity": sensitivity["encoders"],
                          "detectability": detectability["encoders"],
                          "ceilings": detectability["ceilings"]},
                "linear": linear["levels"],
                "runs": {f"{variant}/s{seed}": {
                    "vertex_resolution_m": payload["forms"]["full_input"].get(
                        "vertex_resolution_m"),
                    "module_gain_run_mean_r2": payload["forms"]["full_input"].get(
                        "module_gain_run_mean_r2"),
                    "checkpoint_sha256": payload["checkpoint_sha256"],
                    "wall_seconds": payload.get("wall_seconds")}
                    for (variant, seed), payload in summaries.items()},
                "floors": floors}, overwrite=True)
    (RESULTS / "README.md").write_text(
        "# De-jitter extension (30 Oct 2026)\n\n"
        "**registered: false** — extension `dejitter_2026-10-30`.\n\n"
        "Runbook: `docs/jepa/RUNBOOK_2026-10-30_DEJITTER.md`.\n"
        "Training: six approved L40S runs (across_median, within_median, "
        "within_trigger; two seeds each). CPU Part 1 and the linear check ran on "
        "Deepthought; the median_time random-init floors ran on Titan.\n\n"
        "Contents: `sensitivity.md`, `prescription.md`, `readings.json`, "
        "`summary.json`.\n\n"
        "Checkpoints and large arrays stay under "
        "`/ceph/dwong/jepa/returns/ext_dejitter_20261030/`; paths and hashes are in "
        "`summary.json` (and the training `ext_summary.json` files).\n")
    print(json.dumps(jsonable(readings), indent=1))


def _f(value, digits=4, missing="—"):
    return missing if value is None else f"{float(value):.{digits}f}"


def _table(header, rows):
    rule = "|" + "---|" * len(header)
    return "\n".join(["| " + " | ".join(header) + " |", rule]
                     + ["| " + " | ".join(str(item) for item in row) + " |"
                        for row in rows])


def _readings(sensitivity, detectability, linear, summaries, floors):
    out = {}
    enc = sensitivity["encoders"]
    fix = [enc["fix9000"]["R_jit_phys"], enc["fix9001"]["R_jit_phys"]]
    across = [enc["across9000"]["R_jit_phys"], enc["across9001"]["R_jit_phys"]]
    out["J-1"] = {"fix": fix, "across": across,
                  "ratio": (min(across) / max(fix)) if max(fix) else None,
                  "whitened_across": [enc["across9000"]["whitened_R_jit_phys"],
                                      enc["across9001"]["whitened_R_jit_phys"]],
                  "whitened_fix": [enc["fix9000"]["whitened_R_jit_phys"],
                                   enc["fix9001"]["whitened_R_jit_phys"]]}
    ok = max(fix) > 0 and min(across) >= 1.5 * max(fix)
    out["J-1"]["status"] = "supported" if ok else "not supported"
    init = enc["init9000"]["R_jit_phys"]
    out["J-2"] = {"init9000": init, "fix": fix,
                  "supported": all(value <= 0.5 * init for value in fix)}
    out["J-2"]["status"] = "supported" if out["J-2"]["supported"] else "not supported"
    ceiling = detectability["ceilings"]["raw_pooled"]["delta_robust"]
    across_detect = {label: detectability["encoders"][label]["delta_robust"]
                     for label in ("across9000", "across9001")}
    fix_detect = max(detectability["encoders"][label]["delta_robust"]
                     for label in ("fix9000", "fix9001"))
    init_detect = detectability["encoders"]["init9000"]["delta_robust"]
    out["J-3"] = {"ceiling": ceiling, "status": "supported" if ceiling is not None
                  and ceiling >= 0.2 else "not supported"}
    if out["J-3"]["status"] == "supported":
        out["J-3"]["T-d_recheck"] = {
            "across": across_detect, "fix_max": fix_detect, "init": init_detect,
            "status": "supported" if all(
                value is not None and fix_detect is not None and init_detect is not None
                and value >= fix_detect + 0.1 and value >= init_detect + 0.1
                for value in across_detect.values()) else "not supported"}
    refs = {"across9000": 0.348113310588429, "across9001": 0.333463657,
            "within9000": 0.21925822392091457, "within9001": 0.2042813949606102}
    linear_ok = True
    p1_values = {}
    for name, level in linear["levels"].items():
        for split in ("across_fixed",):
            value = level["splits"][split]["variants"]["median_time"]["nu_star_A"]
            p1_values[f"{name}/{split}"] = value
            linear_ok &= value is not None and value <= 0.05
    out["P-1"] = {"nu_star_A_median_time": p1_values,
                  "status": "supported" if linear_ok else "not supported"}
    vertex = {}
    for (variant, seed), payload in summaries.items():
        vertex[(variant, seed)] = payload["forms"]["full_input"].get(
            "vertex_resolution_m")
    p2 = []
    for seed in (9000, 9001):
        value = vertex.get(("across_median", seed))
        reference = refs[f"across{seed}"]
        p2.append({"seed": seed, "vertex": value, "reference": reference,
                   "ok": value is not None and value <= 0.85 * reference})
    out["P-2"] = {"seeds": p2,
                  "status": "supported" if all(item["ok"] for item in p2)
                  else "not supported"}
    p3 = []
    for seed in (9000, 9001):
        med = vertex.get(("across_median", seed))
        control = vertex.get(("within_trigger", seed))
        p3.append({"seed": seed, "across_median": med, "within_trigger": control,
                   "ok": med is not None and control is not None
                   and med <= 1.1 * control})
    out["P-3"] = {"seeds": p3,
                  "status": "supported" if p3 and all(item["ok"] for item in p3)
                  else ("unclear" if not any(vertex.get(("within_trigger", seed))
                                             for seed in (9000, 9001))
                        else "not supported")}
    p4 = []
    for seed in (9000, 9001):
        med = vertex.get(("within_median", seed))
        control = vertex.get(("within_trigger", seed))
        p4.append({"seed": seed, "within_median": med, "within_trigger": control,
                   "ok": med is not None and control is not None
                   and med <= 1.1 * control})
    out["P-4"] = {"seeds": p4,
                  "status": "supported" if p4 and all(item["ok"] for item in p4)
                  else ("unclear" if not any(vertex.get(("within_trigger", seed))
                                             for seed in (9000, 9001))
                        else "not supported")}
    out["P-5"] = {"status": "descriptive",
                  "floors": floors,
                  "runs": {f"{variant}/s{seed}": payload["forms"]["full_input"]
                           for (variant, seed), payload in summaries.items()}}
    return out


# -------------------------------------------------------------------- main
def parser():
    p = argparse.ArgumentParser(prog="jepa_ext_dejitter")
    sub = p.add_subparsers(dest="mode", required=True)

    def common(sp):
        sp.add_argument("--config", default=CONFIG)
        sp.add_argument("--generator-config", default=GEN_CONFIG)
        sp.add_argument("--arms-config", default=None)
        sp.add_argument("--geometry", default=GEOMETRY)
        sp.add_argument("--data-root", default=DATA_ROOT)
        sp.add_argument("--device", default="cpu")

    s = sub.add_parser("sens-gen"); common(s)
    s = sub.add_parser("sens-encode"); common(s); s.add_argument("--label", required=True)
    s = sub.add_parser("sens-aggregate"); common(s)
    s = sub.add_parser("detect-gen"); common(s)
    s.add_argument("--shard", type=int, required=True); s.add_argument("--shards", type=int, default=8)
    s = sub.add_parser("detect-merge"); common(s); s.add_argument("--shards", type=int, default=8)
    s = sub.add_parser("detect-encode"); common(s); s.add_argument("--label", required=True)
    s = sub.add_parser("detect-aggregate"); common(s)
    s = sub.add_parser("linear"); common(s)
    s = sub.add_parser("train"); common(s)
    s.add_argument("--variant", choices=TRAIN_VARIANTS, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--transform", choices=("none", "median_time"), default=None)
    s.add_argument("--steps", type=int, default=None)
    s.add_argument("--probe-max-events", type=int, default=None)
    s.add_argument("--out", required=True)
    s = sub.add_parser("inspect"); common(s)
    s.add_argument("--cell", choices=("dev", "across", "across_median",
                                      "within_median", "within_trigger"), required=True)
    s.add_argument("--item", action="append", required=True)
    s.add_argument("--transform", choices=("none", "median_time"), default="none")
    s.add_argument("--out", required=True)
    s = sub.add_parser("report"); common(s); s.add_argument("--out", default=None)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode == "train":
        args.device = "auto"
        if args.variant.endswith("_median") and args.transform is None:
            args.transform = "median_time"
        elif args.variant == "within_trigger" and args.transform is None:
            args.transform = "none"
    if args.mode == "sens-gen":
        mode_sens_gen(args)
    elif args.mode == "sens-encode":
        mode_sens_encode(args)
    elif args.mode == "sens-aggregate":
        mode_sens_aggregate(args)
    elif args.mode == "detect-gen":
        mode_detect_gen(args)
    elif args.mode == "detect-merge":
        mode_detect_merge(args)
    elif args.mode == "detect-encode":
        mode_detect_encode(args)
    elif args.mode == "detect-aggregate":
        mode_detect_aggregate(args)
    elif args.mode == "linear":
        mode_linear(args)
    elif args.mode == "train":
        mode_train(args)
    elif args.mode == "inspect":
        mode_inspect(args)
    else:
        mode_report(args)


if __name__ == "__main__":
    main()
