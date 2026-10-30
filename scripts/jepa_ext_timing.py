#!/usr/bin/env python3
"""Unregistered timing extension driver (runbook 2026-10-30, Parts A and B).

Modes
-----
parta      reproduce the M-dose covariance read-outs (S8), verify the pooled
           layout, run the §2.3 variants and the §2.5 alignment on the stored
           `.npz` files.  Writes `<ext>/part_a.json`.
dejitter   regenerate the F1_0.00 and F1_0.05 calibration sets, record the
           readout triggers, verify the regenerated A/B blocks against the
           stored ones, build the exact de-jittered A/B blocks and report
           nu*(A)/nu*(B) for `all` and `time_only`.  Writes per-level `.npz`
           under `<ext>/dejitter/` and `<ext>/dejitter.json`.
partb      generate the 1 000-event F1 s=0.05 jitter data, run the eight frozen
           encoders forward once, and probe delta / mu_t / vertex / log yield
           with and without dark-moved triggers.  Writes `<ext>/part_b/`.
report     aggregate into `results/jepa/20261030_timing/` (attribution.md,
           deep_jitter.md, nu_vs_r_v2.png, readings.json, README.md).

Everything is an extension: ``registered: false``.  No training.  Run from the
repository root as:

    PYTHONPATH=src python -m scripts.jepa_ext_timing parta
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from jepa_noise import ext_timing as et
from jepa_noise import lucid_bombs as lb

REPO = Path(__file__).resolve().parents[1]
EXT = Path("/ceph/dwong/jepa/returns/ext_timing_20261030")
M_DOSE_EXT = Path("/ceph/dwong/jepa/returns/ext_m_dose_20261030")
M_DOSE_RESULTS = REPO / "results/jepa/20261030_m_dose"
TIMING_RESULTS = REPO / "results/jepa/20261030_timing"
CONFIG = "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json"
RUN_CONFIG = "src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json"
ARMS_CONFIG = "src/jepa_noise/configs/lucid_arms_v0.json"
GEOMETRY = "/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json"
DATA_ROOT = "/ceph/dwong/jepa/data/lucid_r6_v2"
SEED = 20261017
SPREAD = 0.05
K = 4
LEVELS = (("F1", 0.00), ("F1", 0.02), ("F1", 0.05), ("F1", 0.10),
          ("F1", 0.20), ("F1", 0.30), ("F1x", 0.50), ("F1x", 0.75),
          ("F1x", 1.00), ("F1x", 1.50))
SPLITS = ("within", "across_fixed")
VARIANTS = ("all", "time_only", "charge_only", "hit_only", "no_time",
            "time_common_removed", "time_hit_removed")
N_CAL = 1000
REPLICATES = 8
N_RUNS = 100
N_EVENTS = 10
R_PARTB = 16


def stem(law, spread):
    return f"{law}_{spread:.2f}"


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
    raw = (json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n").encode()
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


def check_storage(root=EXT):
    root = Path(root).resolve()
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
        raise SystemExit(f"S1: {free} free entries < 30000; write nothing")
    if entries >= 1000 or size >= 1_000_000_000:
        raise SystemExit(f"S1: extension storage {entries} entries / {size} bytes "
                         f"exceeds the 1000-entry / 1 GB budget")
    return {"free_entries": free, "extension_entries": entries,
            "extension_bytes": size}


def check_provenance():
    digest = sha256_file(CONFIG)
    recorded = json.loads(
        (REPO / "results/jepa/20261025_pregrid_checks/provenance.json").read_text())
    expected = recorded["generator_config"]["sha256"]
    if digest != expected:
        raise SystemExit(f"generator config hash mismatch: {digest} != {expected}")
    hashes = json.loads((M_DOSE_RESULTS / "npz_hashes.json").read_text())
    verified = {}
    for name, record in hashes.items():
        path = M_DOSE_EXT / name
        got = sha256_file(path)
        if got != record["sha256"]:
            raise SystemExit(f"M-dose npz hash mismatch for {name}: {got}")
        verified[name] = got
    geometry = lb.load_wcte_geometry(GEOMETRY)
    return {
        "generator_config": {"path": CONFIG, "sha256": digest,
                             "provenance_sha256": expected},
        "m_dose_npz_verified": verified,
        "geometry": {"path": GEOMETRY, "sha256": sha256_file(GEOMETRY),
                     "digest": geometry.digest, "n_pmt": int(geometry.n_pmt),
                     "n_modules": int(geometry.n_modules)},
        "storage_before": check_storage(),
    }


def load_level(law, spread):
    name = stem(law, spread)
    with np.load(M_DOSE_EXT / f"{name}.npz", allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    payload = json.loads((M_DOSE_RESULTS / f"{name}.json").read_text())
    return arrays, payload


def split_blocks(arrays, split):
    S = arrays[f"{split}_S"]
    A = arrays[f"{split}_A"]
    B = arrays[f"{split}_B"]
    null = arrays[f"{split}_null"]
    means = arrays[f"{split}_means"]
    context_dim = int(arrays[f"{split}_context_dim"])
    target_dim = int(arrays[f"{split}_target_dim"])
    blocks = et.blocks_from_joint(S, A, B, context_dim, target_dim)
    return blocks, null, means, context_dim, target_dim


def _rows(payload, split):
    return next(row for row in payload["rows"] if row["split"] == split)


def mode_parta(args):
    provenance = check_provenance()
    geometry = lb.load_wcte_geometry(GEOMETRY)
    from scripts.jepa_ext_m_dose import view_groups
    groups = view_groups(geometry)
    layout = {}
    for split in SPLITS:
        dims = [3 * len(groups[split][0]), 3 * len(groups[split][1])]
        layout[split] = {"context_dim": dims[0], "target_dim": dims[1],
                         "context_groups": len(groups[split][0]),
                         "target_groups": len(groups[split][1])}
    # Layout check on the stored files and on the pooled-index convention.
    for law, spread in LEVELS:
        arrays, _ = load_level(law, spread)
        for split in SPLITS:
            stored = (int(arrays[f"{split}_context_dim"]),
                      int(arrays[f"{split}_target_dim"]))
            expected = (layout[split]["context_dim"], layout[split]["target_dim"])
            if stored != expected:
                raise SystemExit(f"layout mismatch {law} {spread} {split}: "
                                 f"{stored} != {expected}")
    # The M-dose driver's pooled index is group * 3 + f, context groups first.
    probe = np.zeros((1, geometry.n_pmt, 9))
    probe[0, groups["within"][0][3], et.RAW_FEATURE_BASE + et.HIT_OFFSET] = 7.0
    probe[0, groups["within"][1][4], et.RAW_FEATURE_BASE + et.HIT_OFFSET] = 7.0
    from scripts.jepa_ext_m_dose import pool
    pooled_probe = pool(probe, list(groups["within"][0]) + list(groups["within"][1]))
    layout["pooled_index_verified"] = bool(
        abs(pooled_probe[0, 3 * 3 + et.HIT_OFFSET] - 7.0) < 1e-12
        and abs(pooled_probe[0, 3 * (len(groups["within"][0]) + 4) + et.HIT_OFFSET]
                - 7.0) < 1e-12)
    layout["pooled_index"] = "group*3+f with f=0 log_charge, 1 time_100ns, 2 hit; context groups first"
    if not layout["pooled_index_verified"]:
        raise SystemExit("module-pooled layout verification failed")

    reproduction = {}
    variants = {}
    alignment = {}
    worst = 0.0
    for law, spread in LEVELS:
        arrays, payload = load_level(law, spread)
        name = stem(law, spread)
        reproduction[name] = {}
        for split in SPLITS:
            blocks, null, means, context_dim, target_dim = split_blocks(arrays, split)
            metrics = et.pair_metrics(blocks, K, (blocks["act"],
                                                  blocks["bct"] - blocks["act"]))
            row = _rows(payload, split)
            comparisons = {
                "nu_star": (metrics["nu_star_B"], row["nu_star"]),
                "nu_star_A": (metrics["nu_star_A"], row["nu_star_A"]),
                "nu_star_gain": (metrics["nu_star_gain"], row["nu_star_gain"]),
                "sigma_phys": (metrics["sigma_phys"], row["sigma_phys"]),
                "sigma_noise": (metrics["sigma_noise"], row["sigma_noise"]),
                "r": (metrics["r"], row["r"]),
            }
            differences = {key: abs(float(first) - float(second))
                           for key, (first, second) in comparisons.items()}
            worst = max(worst, max(differences.values()))
            reproduction[name][split] = differences
            if max(differences.values()) > 1e-9:
                raise SystemExit(f"S8 reproduction failed for {name} {split}: {differences}")

            variants[f"{name}/{split}"] = {}
            for variant in VARIANTS:
                transformed, Tc, Tt = et.variant_blocks(
                    blocks, variant, S=arrays[f"{split}_S"], means=means,
                    context_dim=context_dim, target_dim=target_dim)
                variant_metrics = et.pair_metrics(
                    transformed, K, (transformed["act"],
                                     transformed["bct"] - transformed["act"]))
                variants[f"{name}/{split}"][variant] = {
                    "context_dim": int(Tc.shape[0]),
                    "target_dim": int(Tt.shape[0]),
                    **variant_metrics,
                }
        if name == "F1_0.00":
            for split in SPLITS:
                blocks, _, means, context_dim, _ = split_blocks(arrays, split)
                alignment[split] = et.alignment_metrics(
                    blocks, means[:, :context_dim])

    payload = {
        "registered": False, "extension": "timing_2026-10-30", "kind": "part_a",
        "layout": layout, "provenance": provenance,
        "reproduction_max_abs_difference": worst,
        "reproduction": reproduction, "variants": variants,
        "alignment": alignment,
        "invocation": "PYTHONPATH=src python -m scripts.jepa_ext_timing parta",
    }
    write_json(EXT / "part_a.json", payload)
    print(f"PART A OK: S8 max difference {worst:.3e}")


def _state_for(geometry, law, spread, run_id, seed=SEED):
    from scripts.jepa_ext_m_dose import state_for
    return state_for(geometry, law, spread, run_id, seed=seed)


def _prepare_generation(geometry, cfg):
    from jepa_noise import point_encoder as pe
    from jepa_noise.lucid_views import ViewMasker
    from scripts.jepa_ext_m_dose import view_groups
    return {
        "readout": lb.make_readout(geometry, cfg),
        "scaling": pe.FeatureScaling.from_geometry(geometry),
        "masker": ViewMasker(geometry, "within", n_min=0),
        "groups": view_groups(geometry),
    }


def _tokenise(geometry, hits, tools):
    from scripts.jepa_ext_m_dose import tokenise_hits
    return tokenise_hits(geometry, hits, tools["scaling"], tools["masker"])


def _pooled(features, split, tools):
    from scripts.jepa_ext_m_dose import pooled
    return pooled(features, split, tools["groups"])


def _trigger(record):
    """trigger_ns, falling back to the source trigger (recorded and reported)."""
    if record.trigger_ns is not None:
        return float(record.trigger_ns), False
    if record.source_trigger_ns is not None:
        return float(record.source_trigger_ns), True
    return None, True


def mode_dejitter(args):
    provenance = check_provenance()
    geometry = lb.load_wcte_geometry(GEOMETRY)
    cfg = lb.load_generator_config(CONFIG)
    from jepa_noise.ext_predictability import resample_noise
    tools = _prepare_generation(geometry, cfg)
    output = {"registered": False, "extension": "timing_2026-10-30",
              "kind": "dejitter", "provenance": provenance, "levels": {}}
    (EXT / "dejitter").mkdir(parents=True, exist_ok=True)
    for law, spread in (("F1", 0.00), ("F1", 0.05)):
        name = stem(law, spread)
        arrays, _ = load_level(law, spread)
        scale = np.sqrt(REPLICATES / (REPLICATES - 1))
        accum = {split: {"A": np.zeros_like(arrays[f"{split}_A"]),
                         "B": np.zeros_like(arrays[f"{split}_B"]),
                         "A_dejitter": np.zeros_like(arrays[f"{split}_A"]),
                         "B_dejitter": np.zeros_like(arrays[f"{split}_B"]),
                         "triggers_A": [], "triggers_B": [],
                         "fallback_A": 0, "fallback_B": 0}
                 for split in SPLITS}
        started = time.monotonic()
        for event in range(N_CAL):
            run_id = f"ext_m_dose/cal/{event:04d}"
            draws = lb.draw_event(geometry, cfg, run_id, event, seed=SEED,
                                  domain="photon")
            fixed_state = _state_for(geometry, law, spread, run_id)
            hits_a, hits_b, triggers_a, triggers_b = [], [], [], []
            fallback_a = fallback_b = 0
            for replicate in range(REPLICATES):
                record_a = tools["readout"](fixed_state, resample_noise(
                    draws, geometry, cfg, replicate, seed=SEED))
                trigger_a, fallback = _trigger(record_a)
                fallback_a += int(fallback)
                hits_a.append(record_a)
                triggers_a.append(trigger_a)
                fresh_seed = int(lb.keyed_rng(
                    SEED, "ext_predictability_state", run_id, event, replicate)
                    .integers(0, 2**64, dtype=np.uint64))
                fresh_state = _state_for(geometry, law, spread, run_id,
                                         seed=fresh_seed)
                record_b = tools["readout"](fresh_state, resample_noise(
                    draws, geometry, cfg, replicate + 1, seed=SEED))
                trigger_b, fallback = _trigger(record_b)
                fallback_b += int(fallback)
                hits_b.append(record_b)
                triggers_b.append(trigger_b)
            if None in triggers_a or None in triggers_b:
                raise SystemExit(f"{name}: a realisation has no trigger at all")
            features_a, _ = _tokenise(geometry, hits_a, tools)
            features_b, _ = _tokenise(geometry, hits_b, tools)
            deltas_a = et.trigger_deltas(triggers_a)
            deltas_b = et.trigger_deltas(triggers_b)
            mean_trigger_a = float(np.mean(triggers_a))
            mean_trigger_b = float(np.mean(triggers_b))
            for split in SPLITS:
                xa = _pooled(features_a, split, tools)
                xb = _pooled(features_b, split, tools)
                ra = (xa - xa.mean(axis=0)) * scale
                rb = (xb - xb.mean(axis=0)) * scale
                accum[split]["A"] += ra.T @ ra / REPLICATES
                accum[split]["B"] += rb.T @ rb / REPLICATES
                xa_dejitter = et.dejitter_pooled(xa, deltas_a)
                xb_dejitter = et.dejitter_pooled(xb, deltas_b)
                ra = (xa_dejitter - xa_dejitter.mean(axis=0)) * scale
                rb = (xb_dejitter - xb_dejitter.mean(axis=0)) * scale
                accum[split]["A_dejitter"] += ra.T @ ra / REPLICATES
                accum[split]["B_dejitter"] += rb.T @ rb / REPLICATES
                accum[split]["triggers_A"].append(mean_trigger_a)
                accum[split]["triggers_B"].append(mean_trigger_b)
            accum[split]["fallback_A"] += fallback_a
            accum[split]["fallback_B"] += fallback_b
            if event % 100 == 0:
                print(f"dejitter {name}: {event}/{N_CAL}", flush=True)

        level = {"wall_seconds": time.monotonic() - started, "splits": {}}
        arrays_out = {}
        for split in SPLITS:
            context_dim = int(arrays[f"{split}_context_dim"])
            target_dim = int(arrays[f"{split}_target_dim"])
            for key in ("A", "B", "A_dejitter", "B_dejitter"):
                accum[split][key] /= N_CAL
            stored_a = np.asarray(arrays[f"{split}_A"], dtype=np.float64)
            stored_b = np.asarray(arrays[f"{split}_B"], dtype=np.float64)
            difference_a = float(np.max(np.abs(accum[split]["A"] - stored_a)))
            difference_b = float(np.max(np.abs(accum[split]["B"] - stored_b)))
            if difference_a > 1e-9 or difference_b > 1e-9:
                raise SystemExit(f"{name} {split}: regenerated A/B mismatch "
                                 f"{difference_a:.3e} / {difference_b:.3e}")
            S = arrays[f"{split}_S"]
            A_dej = accum[split]["A_dejitter"]
            B_dej = accum[split]["B_dejitter"]
            blocks = et.blocks_from_joint(S, A_dej, B_dej, context_dim, target_dim)
            result = {"regenerated_A_max_abs_difference": difference_a,
                      "regenerated_B_max_abs_difference": difference_b,
                      "fallback_A": int(accum[split]["fallback_A"]),
                      "fallback_B": int(accum[split]["fallback_B"]),
                      "variants": {}}
            for variant in ("all", "time_only"):
                transformed, _, _ = et.variant_blocks(
                    blocks, variant, S=S, means=arrays[f"{split}_means"],
                    context_dim=context_dim, target_dim=target_dim)
                result["variants"][variant] = et.pair_metrics(transformed, K)
            level["splits"][split] = result
            arrays_out[f"{split}_A_dejitter"] = A_dej
            arrays_out[f"{split}_B_dejitter"] = B_dej
            arrays_out[f"{split}_mean_trigger_A"] = np.asarray(
                accum[split]["triggers_A"])
            arrays_out[f"{split}_mean_trigger_B"] = np.asarray(
                accum[split]["triggers_B"])
        path = EXT / "dejitter" / f"{name}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays_out)
        level["npz"] = str(path)
        level["npz_sha256"] = sha256_file(path)
        output["levels"][name] = level
    write_json(EXT / "dejitter.json", output)
    print("DEJITTER OK")


# ------------------------------------------------------------------- Part B
ENCODERS = (
    {"label": "init9000", "arm": "sigreg_jepa", "source": "init", "seed": 9000,
     "cell": "dev", "vertex_m": 0.160904},
    {"label": "fix9000", "arm": "sigreg_jepa", "seed": 9000, "cell": "dev",
     "vertex_m": 0.21925822392091457},
    {"label": "fix9001", "arm": "sigreg_jepa", "seed": 9001, "cell": "dev",
     "vertex_m": 0.2042813949606102},
    {"label": "mae9000", "arm": "mae_mse", "seed": 9000, "cell": "dev",
     "vertex_m": 0.10494918764937129},
    {"label": "ijepa9000", "arm": "ijepa_ema", "seed": 9000, "cell": "dev",
     "vertex_m": 0.209149763},
    {"label": "ijepa9001", "arm": "ijepa_ema", "seed": 9001, "cell": "dev",
     "vertex_m": 0.194991722},
    {"label": "across9000", "arm": "sigreg_jepa", "seed": 9000, "cell": "across",
     "vertex_m": 0.348113310588429},
    {"label": "across9001", "arm": "sigreg_jepa", "seed": 9001, "cell": "across",
     "vertex_m": 0.333463657},
)


def _checkpoint_specs():
    specs = {item["label"]: dict(item) for item in ENCODERS}
    for label in ("fix9000", "fix9001", "mae9000", "ijepa9000", "ijepa9001"):
        record = json.loads(
            (REPO / "results/jepa/20261029_predictability/probes"
             / f"{label}.json").read_text())
        specs[label]["source"] = record["checkpoint"]
        got = sha256_file(record["checkpoint"])
        if got != record["checkpoint_sha256"]:
            raise SystemExit(f"{label}: checkpoint sha256 mismatch {got}")
        specs[label]["checkpoint_sha256"] = got
        specs[label]["checkpoint_verification"] = "probes json sha256"
    sums = {}
    for line in (REPO / "results/jepa/20261029_d2_partial/SHA256SUMS").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1]] = parts[0]
    for label in ("across9000", "across9001"):
        seed = specs[label]["seed"]
        root = ("/ceph/dwong/jepa/returns/ext_pregrid_20261025/train/"
                f"across_s{seed}/runs/ext_dev_s005_across__sigreg_jepa__s{seed}")
        summary = f"{root}/ext_summary.json"
        relative = summary.removeprefix("/ceph/dwong/jepa/")
        expected = sums.get(relative)
        got_summary = sha256_file(summary)
        if expected is None or got_summary != expected:
            raise SystemExit(f"{label}: ext_summary sha256 mismatch "
                             f"{got_summary} != {expected}")
        checkpoint = f"{root}/final.pt"
        specs[label]["source"] = checkpoint
        specs[label]["checkpoint_sha256"] = sha256_file(checkpoint)
        specs[label]["checkpoint_verification"] = "ext_summary sha256 vs SHA256SUMS"
    return specs


def _partb_targets(geometry, cfg):
    from jepa_noise.ext_predictability import resample_noise
    tools = _prepare_generation(geometry, cfg)
    n_events = N_RUNS * N_EVENTS
    triggers = np.full((n_events, R_PARTB), np.nan)
    moved = np.zeros((n_events, R_PARTB), dtype=bool)
    vertex = np.empty((n_events, 3))
    log_yield = np.empty(n_events)
    emission = np.empty(n_events)
    features = np.empty((n_events, geometry.n_pmt, 9), dtype=np.float32)
    cursor = 0
    invalid = 0
    for run in range(N_RUNS):
        run_id = f"ext_m_dose/evt/{run:04d}"
        state = _state_for(geometry, "F1", SPREAD, run_id)
        draws_list = [lb.draw_event(geometry, cfg, run_id,
                                    run * N_EVENTS + event, seed=SEED,
                                    domain="photon")
                      for event in range(N_EVENTS)]
        zero_hits = []
        for event, draws in enumerate(draws_list):
            records = [tools["readout"](state, resample_noise(
                draws, geometry, cfg, replicate, seed=SEED))
                for replicate in range(R_PARTB)]
            for replicate, record in enumerate(records):
                trigger, fallback = _trigger(record)
                triggers[cursor + event, replicate] = (
                    np.nan if trigger is None else trigger)
                moved[cursor + event, replicate] = bool(
                    record.source_trigger_ns is not None
                    and (record.trigger_ns is None or
                         abs(record.trigger_ns - record.source_trigger_ns) > 1e-9))
            zero_hits.append(records[0])
            vertex[cursor + event] = draws.vertex_m
            log_yield[cursor + event] = np.log(draws.yield_photons)
            emission[cursor + event] = draws.emission_offset_ns
        batch_features, _ = _tokenise(geometry, zero_hits, tools)
        features[cursor:cursor + N_EVENTS] = batch_features.astype(np.float32)
        bad = ~np.isfinite(triggers[cursor:cursor + N_EVENTS]).all(axis=1)
        invalid += int(np.sum(bad))
        cursor += N_EVENTS
        if run % 10 == 0:
            print(f"partb data: run {run}/{N_RUNS}", flush=True)
    valid = np.isfinite(triggers).all(axis=1)
    delta = triggers[:, 0] - triggers.mean(axis=1)
    mu_t = triggers.mean(axis=1) - emission
    return {"features": features, "triggers": triggers, "moved": moved,
            "valid": valid, "invalid_events": invalid, "delta": delta,
            "mu_t": mu_t, "vertex": vertex, "log_yield": log_yield,
            "dark_moved_r0": moved[:, 0]}


def _ridge(xtr, ytr, xte, yte):
    from scripts.jepa_pregrid_checks import ridge_r2
    return ridge_r2(xtr, ytr, xte, yte)


def _probe_set(vectors, targets, train_mask, test_mask):
    out = {}
    for name in ("delta", "mu_t", "log_yield"):
        result = _ridge(vectors[train_mask], targets[name][train_mask],
                        vectors[test_mask], targets[name][test_mask])
        out[name] = result["r2"]
    vertex_scores = []
    for coordinate in range(3):
        result = _ridge(vectors[train_mask],
                        targets["vertex"][train_mask, coordinate],
                        vectors[test_mask],
                        targets["vertex"][test_mask, coordinate])
        vertex_scores.append(result["r2"])
    finite = [value for value in vertex_scores if value is not None]
    out["vertex"] = float(np.mean(finite)) if finite else None
    return out


def _encode(model, features, chunk=64):
    import torch
    vectors = []
    with torch.no_grad():
        for start in range(0, len(features), chunk):
            batch = torch.as_tensor(features[start:start + chunk])
            tokens = model.encode(batch)
            vectors.append(tokens.mean(dim=1).cpu().numpy())
    return np.concatenate(vectors).astype(np.float64)


def _raw_ceilings(features, targets, valid, dark_moved):
    from scripts.jepa_ext_m_dose import pooled, view_groups
    geometry = lb.load_wcte_geometry(GEOMETRY)
    groups = view_groups(geometry)
    pooled_features = pooled(features.astype(np.float64), "within", groups)
    hit = features[..., et.RAW_FEATURE_BASE + et.HIT_OFFSET] > 0
    time = features[..., et.RAW_FEATURE_BASE + et.TIME_OFFSET]
    mean_hit_time = np.where(hit, time, 0.0).sum(axis=1) / np.maximum(hit.sum(axis=1), 1)
    run_ids = np.arange(len(features)) // N_EVENTS
    result = {"raw_pooled": {}, "hit_token_mean_time": {}}
    for label, vectors in (("raw_pooled", pooled_features),
                           ("hit_token_mean_time", mean_hit_time[:, None])):
        for subset, mask in (("all", valid),
                             ("without_dark_moved", valid & ~dark_moved)):
            train = mask & (run_ids < 50)
            test = mask & (run_ids >= 50)
            result[label][subset] = _probe_set(vectors, targets, train, test)
    return result


def mode_partb(args):
    provenance = check_provenance()
    import torch
    from jepa_noise import lucid_arms as la
    from jepa_noise import lucid_run as lr
    from scripts import jepa_pregrid_checks as pregrid
    geometry = lb.load_wcte_geometry(GEOMETRY)
    cfg = lb.load_generator_config(CONFIG)
    config = lr.load_run_config(RUN_CONFIG)
    arms_config = la.load_config(ARMS_CONFIG)
    device = lr.configure_runtime("cpu")
    data = _partb_targets(geometry, cfg)
    print(f"partb: invalid events {data['invalid_events']}, dark-moved r=0 "
          f"{int(data['dark_moved_r0'].sum())}", flush=True)
    targets = {"delta": data["delta"], "mu_t": data["mu_t"],
               "vertex": data["vertex"], "log_yield": data["log_yield"]}
    valid = data["valid"]
    dark_moved = data["dark_moved_r0"]
    run_ids = np.arange(len(valid)) // N_EVENTS
    masks = {"all": valid,
             "without_dark_moved": valid & ~dark_moved}
    specs = _checkpoint_specs()
    dev_cell = lr.dev_cell(config)
    across_cell = pregrid.variant_cell(config, "across")
    output = {"registered": False, "extension": "timing_2026-10-30",
              "kind": "part_b", "provenance": provenance,
              "data": {"n_events": int(len(valid)),
                       "invalid_events": int(data["invalid_events"]),
                       "dark_moved_r0": int(dark_moved.sum()),
                       "spread": SPREAD, "replicates": R_PARTB},
              "encoders": {}, "ceilings": {},
              "target_sd": {name: float(np.std(values[valid]))
                            for name, values in targets.items() if name != "vertex"}}
    # Store the event vectors and targets as the extension's large arrays.
    (EXT / "part_b").mkdir(parents=True, exist_ok=True)
    for label in ("init9000", "fix9000", "fix9001", "mae9000", "ijepa9000",
                  "ijepa9001", "across9000", "across9001"):
        spec = specs[label]
        cell = across_cell if spec["cell"] == "across" else dev_cell
        model, beta = pregrid.build_loaded(
            spec["arm"], cell, geometry, arms_config, config, cfg,
            Path(DATA_ROOT), spec["source"], spec["seed"])
        model = getattr(model, "jepa", model).to(device).eval()
        vectors = _encode(model, data["features"])
        blocked = {"train": 0}
        enc = {"arm": spec["arm"], "cell": spec["cell"],
               "checkpoint": spec["source"],
               "checkpoint_sha256": spec.get("checkpoint_sha256"),
               "checkpoint_verification": spec.get("checkpoint_verification"),
               "vertex_resolution_m": spec["vertex_m"], "probes": {}}
        for subset, mask in masks.items():
            train = mask & (run_ids < 50)
            test = mask & (run_ids >= 50)
            if train.sum() < 10 or test.sum() < 10:
                enc["probes"][subset] = {"status": "too_few_rows",
                                         "n_train": int(train.sum()),
                                         "n_test": int(test.sum())}
                continue
            enc["probes"][subset] = _probe_set(vectors, targets, train, test)
        output["encoders"][label] = enc
        np.savez_compressed(EXT / "part_b" / f"{label}.npz",
                            event_vectors=vectors.astype(np.float32),
                            valid=valid, dark_moved_r0=dark_moved,
                            delta=data["delta"], mu_t=data["mu_t"],
                            vertex=data["vertex"], log_yield=data["log_yield"])
        print(f"partb: {label} encoded", flush=True)
    output["ceilings"] = _raw_ceilings(data["features"], targets, valid, dark_moved)
    write_json(EXT / "part_b.json", output)
    print("PART B OK")




# --------------------------------------------------------------- reporting
def _fmt(value, digits=4, missing="—"):
    if value is None:
        return missing
    return f"{float(value):.{digits}f}"


def _table(header, rows):
    rule = "|" + "---|" * len(header)
    return "\n".join(["| " + " | ".join(header) + " |", rule]
                      + ["| " + " | ".join(row) + " |" for row in rows])


def _figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    toy = json.loads((M_DOSE_RESULTS / "toy.json").read_text())["rows"]
    within, across = [], []
    for law, spread in LEVELS:
        payload = json.loads((M_DOSE_RESULTS / f"{stem(law, spread)}.json").read_text())
        within.append(_rows(payload, "within"))
        across.append(_rows(payload, "across_fixed"))
    floors = [row["nu_star_null"] for row in within + across]
    band = float(np.mean(floors) + 2.0 * np.std(floors, ddof=1))
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), dpi=150)
    for axis, key, label in ((axes[0], "r", "r_k = sigma_1^noise / sigma_k^phys"),
                             (axes[1], "r_s", "r_k^S")):
        axis.axhspan(0.0, band, color="0.85", zorder=0,
                     label=f"null floor band (mean+2SD = {band:.3f})")
        axis.plot([row["r"] if key == "r" else row["r_s"] for row in toy],
                  [row["nu_star"] for row in toy], "o-", color="tab:blue",
                  label="toy (population)")
        axis.plot([row[key] for row in within], [row["nu_star"] for row in within],
                  "s:", color="tab:orange", label="LUCiD within")
        axis.plot([row[key] for row in across], [row["nu_star"] for row in across],
                  "^:", color="tab:green", label="LUCiD across_fixed")
        axis.axvline(1.0, color="0.4", linestyle="--", linewidth=1)
        axis.set_xlabel(label)
        axis.set_ylabel("nu* = max_j<=k nu_(j)  (N_B)")
        axis.set_ylim(bottom=-0.02)
        axis.legend(fontsize=8, loc="upper left")
    axes[0].set_title("M dose: nuisance share vs the ratio (nu_vs_r_v2)")
    axes[1].set_title("Same data against r^S")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return {"floor_band": band,
            "floor_band_definition": "mean + 2 SD of the 20 LUCiD split-level nu*_null values",
            "n_floor_values": len(floors)}


def _readings(part_a, dejitter, part_b):
    def variant(level, split, name):
        return part_a["variants"][f"{level}/{split}"][name]

    across0 = variant("F1_0.00", "across_fixed", "no_time")
    across0_all = variant("F1_0.00", "across_fixed", "all")
    within0_all = variant("F1_0.00", "within", "all")
    within0_no_time = variant("F1_0.00", "within", "no_time")
    within0_common = variant("F1_0.00", "within", "time_common_removed")
    within0_time = variant("F1_0.00", "within", "time_only")

    def dejitter_value(level, split, name, key):
        return dejitter["levels"][level]["splits"][split]["variants"][name][key]

    support = {}

    def status(value, rule):
        return "supported" if rule else "not supported"

    support["T-a"] = {
        "rule": "across_fixed, s=0: nu*(A) <= 0.1 in no_time",
        "all_nu_star_A": across0_all["nu_star_A"],
        "no_time_nu_star_A": across0["nu_star_A"],
        "status": status(None, across0["nu_star_A"] <= 0.1),
    }
    t_b_values = {
        f"{level}/{split}/{name}": dejitter_value(level, split, name, "nu_star_A")
        for level in ("F1_0.00", "F1_0.05") for split in ("across_fixed",)
        for name in ("all", "time_only")
    }
    support["T-b"] = {
        "rule": "across_fixed at s=0 and s=0.05: nu*(A_dejitter) <= 0.1 in all and time_only",
        "values": t_b_values,
        "status": status(None, all(value <= 0.1 for value in t_b_values.values())),
    }
    support["T-c"] = {
        "rule": "within, s=0: nu*(A) <= 0.05 in all, no_time, time_common_removed",
        "all_nu_star_A": within0_all["nu_star_A"],
        "no_time_nu_star_A": within0_no_time["nu_star_A"],
        "time_common_removed_nu_star_A": within0_common["nu_star_A"],
        "time_only_nu_star_A_descriptive": within0_time["nu_star_A"],
        "status": status(None, max(within0_all["nu_star_A"],
                                   within0_no_time["nu_star_A"],
                                   within0_common["nu_star_A"]) <= 0.05),
    }
    support["M-dose correction recorded by the author"] = {
        "across_fixed_nu_star_at_s0": across0_all["nu_star"],
        "across_fixed_nu_star_A_at_s0": across0_all["nu_star_A"],
        "across_fixed_nu_star_gain_at_s0": across0_all["nu_star_gain"],
        "note": "the author's correction is confirmed on the stored data",
    }
    if part_b is None:
        support["T-d"] = {"status": "not_run", "rule": "Part B pending"}
        support["T-e"] = {"status": "not_run", "rule": "Part B pending"}
        return support
    encoders = part_b["encoders"]
    without = {label: record["probes"]["without_dark_moved"]
               for label, record in encoders.items()}
    init = without["init9000"]["delta"]
    across = {label: record["delta"] for label, record in without.items()
              if label.startswith("across")}
    # label_starts is defined below; simple local max over fix seeds here.
    fix_values = [without[label]["delta"] for label in ("fix9000", "fix9001")
                  if without[label].get("delta") is not None]
    fix_max = max(fix_values) if fix_values else None
    ceiling = part_b["ceilings"]["raw_pooled"]["without_dark_moved"]["delta"]
    t_d_ok = all(value is not None and fix_max is not None and init is not None
                 and value >= fix_max + 0.1 and value >= init + 0.1
                 for value in across.values())
    if ceiling is not None and ceiling < 0.2:
        t_d_status = "unclear"
    else:
        t_d_status = "supported" if t_d_ok else "not supported"
    support["T-d"] = {
        "rule": "both across seeds: R2(delta) >= max(fix) + 0.1 and >= init9000 + 0.1; "
                "unclear if raw-input ceiling R2(delta) < 0.2",
        "across_R2_delta": across, "fix_max": fix_max, "init9000": init,
        "raw_ceiling": ceiling, "status": t_d_status,
    }
    from scipy.stats import spearmanr
    pairs = [(record["probes"]["without_dark_moved"]["delta"],
              record["vertex_resolution_m"])
             for record in encoders.values()
             if record["probes"]["without_dark_moved"].get("delta") is not None]
    rho = spearmanr([pair[0] for pair in pairs], [pair[1] for pair in pairs]).statistic
    support["T-e"] = {
        "rule": "descriptive: Spearman(R2(delta), Part C/D vertex resolution) over all encoders",
        "spearman": float(rho) if np.isfinite(rho) else None,
        "pairs": pairs, "status": "descriptive",
    }
    return support


def mode_report(args):
    part_a = json.loads((EXT / "part_a.json").read_text())
    dejitter = json.loads((EXT / "dejitter.json").read_text())
    part_b_path = EXT / "part_b.json"
    part_b = json.loads(part_b_path.read_text()) if part_b_path.exists() else None
    TIMING_RESULTS.mkdir(parents=True, exist_ok=True)
    figure_meta = _figure(TIMING_RESULTS / "nu_vs_r_v2.png")
    readings = _readings(part_a, dejitter, part_b)

    rows = []
    for law, spread in LEVELS:
        name = stem(law, spread)
        for split in SPLITS:
            for variant in VARIANTS:
                value = part_a["variants"][f"{name}/{split}"][variant]
                rows.append([law, f"{spread:.2f}", split, variant,
                             _fmt(value["nu_star_B"]), _fmt(value["nu_star_A"]),
                             _fmt(value["nu_star_gain"]),
                             _fmt(value["sigma_phys"]), _fmt(value["sigma_noise"]),
                             _fmt(value["r"]), str(value["argmax_B"])])
    attribution = [
        "# Timing extension — Part A attribution (registered: false)", "",
        "## S8 reproduction", "",
        f"Maximum absolute difference from `results/jepa/20261030_m_dose/*.json`: "
        f"`{part_a['reproduction_max_abs_difference']:.3e}` (tolerance 1e-9).", "",
        "## Layout (verified against the code)", "",
        "`" + part_a["layout"]["pooled_index"] + "`", "",
        f"- within: {part_a['layout']['within']['context_dim']} + "
        f"{part_a['layout']['within']['target_dim']}",
        f"- across_fixed: {part_a['layout']['across_fixed']['context_dim']} + "
        f"{part_a['layout']['across_fixed']['target_dim']}", "",
        "## Variants (nu* under N_B / N_A / N_gain, sigma_k^phys, sigma_1^noise, r, argmax pair)", "",
        _table(["law", "s", "split", "variant", "nu*(B)", "nu*(A)", "nu*(gain)",
                "sigma_k^phys", "sigma_1^noise", "r", "argmax"], rows), "",
        "## Alignment at s = 0 (A blocks)", "",
        _table(["split", "cos(p, 1_time)", "cos(p, m_h time)"],
               [[split, _fmt(part_a["alignment"][split]["cos_time_ones"], 4),
                 _fmt(part_a["alignment"][split]["cos_time_hit"], 4)]
                for split in SPLITS]), "",
        "## Exact de-jitter (regenerated calibration sets)", "",
    ]
    dejitter_rows = []
    for name, level in dejitter["levels"].items():
        for split, result in level["splits"].items():
            for variant in ("all", "time_only"):
                value = result["variants"][variant]
                dejitter_rows.append([name, split, variant,
                                      _fmt(value["nu_star_A"]),
                                      _fmt(value["nu_star_B"])])
    attribution += [_table(["level", "split", "variant", "nu*(A_dejitter)",
                            "nu*(B_dejitter)"], dejitter_rows), "",
                    "Regeneration checks (max abs difference from stored blocks): "
                    + "; ".join(
                        f"{name}/{split} A={result['regenerated_A_max_abs_difference']:.1e} "
                        f"B={result['regenerated_B_max_abs_difference']:.1e}"
                        for name, level in dejitter["levels"].items()
                        for split, result in level["splits"].items()), ""]
    (TIMING_RESULTS / "attribution.md").write_text("\n".join(attribution))

    deep_rows = []
    if part_b is None:
        (TIMING_RESULTS / "deep_jitter.md").write_text(
            "# Timing extension — Part B deep encoders (registered: false)\n\n"
            "Part B has not run yet.\n")
        part_b = {"encoders": {}, "ceilings": {}, "data": None,
                  "ceilings": {"raw_pooled": {}, "hit_token_mean_time": {}}}
    for label, record in part_b["encoders"].items():
        for subset in ("all", "without_dark_moved"):
            probe = record["probes"][subset]
            deep_rows.append([label, record["cell"], subset,
                              _fmt(probe.get("delta")), _fmt(probe.get("mu_t")),
                              _fmt(probe.get("vertex")),
                              _fmt(probe.get("log_yield")),
                              _fmt(record["vertex_resolution_m"])])
    ceiling_rows = []
    for label in ("raw_pooled", "hit_token_mean_time"):
        for subset, probe in part_b["ceilings"][label].items():
            ceiling_rows.append([label, subset, _fmt(probe.get("delta")),
                                 _fmt(probe.get("mu_t")),
                                 _fmt(probe.get("vertex")),
                                 _fmt(probe.get("log_yield"))])
    deep = [
        "# Timing extension — Part B deep encoders (registered: false)", "",
        f"Events: {part_b['data']['n_events']} F1 s=0.05 photon events "
        f"(runs 0-99 x 10), {part_b['data']['replicates']} scheme-A realisations each; "
        f"invalid trigger events: {part_b['data']['invalid_events']}; "
        f"dark-moved r=0 triggers: {part_b['data']['dark_moved_r0']}.", "",
        "## Probes (full-input encoder tokens, ridge train runs 0-49 / test 50-99)", "",
        _table(["encoder", "cell", "subset", "R2(delta)", "R2(mu_t)", "R2(vertex)",
                "R2(log yield)", "vertex res (m)"], deep_rows), "",
        "## Ceilings and floor rows", "",
        _table(["row", "subset", "R2(delta)", "R2(mu_t)", "R2(vertex)",
                "R2(log yield)"], ceiling_rows), "",
    ]
    (TIMING_RESULTS / "deep_jitter.md").write_text("\n".join(deep))
    write_json(TIMING_RESULTS / "readings.json", readings, overwrite=True)
    write_json(TIMING_RESULTS / "summary.json",
               {"registered": False, "extension": "timing_2026-10-30",
                "provenance": part_a["provenance"],
                "figure": figure_meta,
                "part_b_data": part_b["data"]}, overwrite=True)
    (TIMING_RESULTS / "README.md").write_text(
        "# Timing extension (30 Oct 2026)\n\n"
        "**registered: false** — extension `timing_2026-10-30`.\n\n"
        "Runbook: `docs/jepa/RUNBOOK_2026-10-30_TIMING.md`.\n"
        "No training; Part B runs frozen checkpoints forward only; CPU only.\n\n"
        "Contents: `attribution.md` (Part A), `deep_jitter.md` (Part B), "
        "`nu_vs_r_v2.png`, `readings.json`, `summary.json`.\n\n"
        "Large arrays (de-jitter blocks, event vectors) stay under "
        "`/ceph/dwong/jepa/returns/ext_timing_20261030/`; their hashes are in "
        "`summary.json`/`dejitter.json`.\n")
    print(json.dumps(jsonable(readings), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("parta", "dejitter", "partb", "report"))
    args = parser.parse_args(argv)
    if args.mode == "parta":
        mode_parta(args)
    elif args.mode == "dejitter":
        mode_dejitter(args)
    elif args.mode == "partb":
        mode_partb(args)
    else:
        mode_report(args)


if __name__ == "__main__":
    main()
