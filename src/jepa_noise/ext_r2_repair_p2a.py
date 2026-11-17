"""Evaluate split-penalty Ridge on the original S4 inputs and the repair grid."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from . import ext_p2 as p2
from . import ext_p2a as p2a
from . import ext_repair as er
from . import lucid_bombs as lb
from . import r2_generator as gen
from . import r2_geometry as rg
from . import r2_store as rs
from .ext_r2_repair_diagnostics import EXTENSION, ROOT, write_json
from .ext_r2_repair_probes import checkpoint_candidates, dataset_path, make_model, safe_output
from .lucid_views import ViewMasker
from .p2a_ceiling_repair import fit_block_ridge, TOLERANCE_M

CASES = (("sigreg_jepa", 9000, "fix9000"), ("sigreg_jepa", 9001, "fix9001"),
         ("target_posfree", 9000, "target_posfree9000"), ("target_posfree", 9001, "target_posfree9001"),
         ("mae_mse", 9000, "mae_within_trigger9000"), ("mae_mse", 9001, "mae_within_trigger9001"))
D1 = (("N", None), ("D1_s015", 0.15), ("D1_s030", 0.30))
D2 = (("N", 0.0), ("D2_j25", 25.0), ("D2_j40", 40.0))


def _mean_3d(prediction, truth):
    return float(np.linalg.norm(prediction - truth, axis=1).mean())


def _original_r2(data, frozen):
    """Reproduce the original P2a R2 nuisance regression + frozen nominal probe."""
    n_train, n_test = data["n_train"], data["n_test"]
    train, test = data["c_train"], data["c_test"]
    hbar, geometry, condition = data["hbar"], data["geometry"], data["condition"]
    dead = p2a._dead_by_run(train, test, condition, geometry)
    u_train = p2.r2_design(train, condition, dead, hbar, geometry)
    u_test = p2.r2_design(test, condition, dead, hbar, geometry)
    i_nt, i_ct, g_t = p2._align_subset(n_train, train)
    i_ne, i_ce, g_e = p2._align_subset(n_test, test)
    z_train = np.asarray(train["events"], float)
    z_test = np.asarray(test["events"], float)
    reg = er.equivariance_fit(
        u_train, z_train - np.asarray(n_train["events"], float)[i_nt],
        u_test, z_test - np.asarray(n_test["events"], float)[i_ne],
        run_ids_train=g_t, run_ids_test=g_e,
        kind="occupancy" if condition.kind == "dead" else "global",
    )
    test_corrected = dict(test, events=z_test - u_test @ np.asarray(er.raw_map(reg), float).T)
    err_r2 = p2a.frozen_errors(n_train, test_corrected, frozen)[i_ce]
    return err_r2, i_ce, g_e


def audit_original(label):
    geometry = rg.load_r2_geometry("wcte")
    drift = p2.build_drift_conditions()
    cache = p2.n_embedding_cache(p2a.CORE_MODELS[0])
    n0 = p2.cache_rows(cache, er.embedding_cache_key("train", "N"))
    data = dict(geometry=geometry, drift=drift, raw={}, hbar=float(np.mean(n0["n_hit"])),
                infos={(name, role): p2.load_transform_info(p2.EXT_ROOT, role, name)
                       for name in p2.CONDITIONS for role in ("train", "test")})
    committed_path = lb._repo_root() / "results/jepa/20261111_p2a/p2a.json"
    committed = json.loads(committed_path.read_text())["models"][label]["cells"]
    rows = []
    for exact in (False, True):
        loaded = p2a.model_cells(data, label, exact=exact)
        for name, cell in loaded["cells"].items():
            prior = committed[name]
            if exact:
                prior = prior.get("exact_subset_reading")
            if not prior or "ceiling" not in prior:
                continue
            train, test = cell["c_train"], cell["c_test"]
            condition = cell["condition"]
            dead = p2a._dead_by_run(train, test, condition, geometry)
            side_train = p2a.fair_design(train, condition, dead, data["hbar"], geometry)
            side_test = p2a.fair_design(test, condition, dead, data["hbar"], geometry)
            r2_errors, indices, runs = _original_r2(cell, loaded["frozen"])
            r2_value = float(r2_errors.mean())
            expected = prior["ceiling"]["v_R2_recomputed_m"]
            if not np.isclose(r2_value, expected, atol=1e-9, rtol=0):
                raise ValueError(f"{label}/{name}/{exact}: original R2 score changed")
            split = fit_block_ridge(train["events"], side_train, train["vertex"], train["run_ids"])
            pred = split.predict(test["events"], side_test)[indices]
            err = np.linalg.norm(pred - test["vertex"][indices], axis=1)
            diff = p2a.ceiling_gap_ci(r2_errors, err, runs)
            gap = r2_value - float(err.mean())
            rows.append({"model": label, "condition": name, "stratum": "exact" if exact else "full",
                         "n": len(err), "original_r2_m": r2_value, "original_r3u_m": prior["v_R3u_m"],
                         "original_s4_violation": prior["ceiling"]["violation"],
                         "split_r3u_m": float(err.mean()), "split_fit": split.settings(),
                         "r2_minus_split_m": gap, "ci95_m": diff["ci95"],
                         "s4_violation": bool(gap < -TOLERANCE_M and diff["ci95"][1] < 0.0)})
    print(f"original {label}: {len(rows)} cells", flush=True)
    return rows


@torch.inference_mode()
def _grid_features(model, geometry, scaling, cohort, gain_s, jitter, *, device):
    masker = ViewMasker(geometry, "within", n_min=0, seed=0)
    attr = ["base", "contrast", "run_ids", "event_ids", "vertex"]
    accum = {name: [] for name in attr}
    specs = gen.cohort_runs(cohort.name)
    for i, run_id in enumerate(cohort.run_ids):
        store = cohort.run_store(i)
        truth = cohort.run_truth(i)
        indices = np.flatnonzero(truth["domain"].astype(str) == lb.DOMAIN_PHOTON)
        old_state = specs[i].state(geometry)
        if gain_s is None:
            gain_log = np.zeros(geometry.n_pmt)
        else:
            drift = replace(specs[i], setting=gain_s).state(geometry)
            gain_log = np.log(drift.gains / old_state.gains)
        for start in range(0, len(indices), 32):
            index = indices[start:start + 32]
            events = truth["event_ids"][index]
            batch = rs.tokenise_sparse(store, geometry, event_ids=events, scaling=scaling,
                                       masker=masker)
            if not np.array_equal(events, batch.event_ids):
                raise ValueError(f"{run_id}: grid event keys mismatch")
            x = batch.features.to(device)
            hit = x[..., 8]
            x[..., 6] += hit * torch.as_tensor(gain_log, dtype=x.dtype, device=device)
            time_shift = [lb.keyed_rng(20261117, "r2repair/p2a/time_zero", run_id, int(e)).normal()
                          for e in events]
            x[..., 7] += (torch.as_tensor(time_shift, dtype=x.dtype, device=device)[:, None]
                          * (jitter / scaling.time_scale_ns) * hit)
            hidden = batch.target.to(device)
            z_a = model.jepa.encode(x, hidden_mask=hidden)
            z_b = model.jepa.encode(x, hidden_mask=~hidden)
            a = (z_a * (~hidden)[..., None]).sum(1) / (~hidden).sum(1, keepdim=True)
            b = (z_b * hidden[..., None]).sum(1) / hidden.sum(1, keepdim=True)
            accum["base"].append(a.cpu().numpy())
            accum["contrast"].append((a - b).cpu().numpy())
            accum["event_ids"].append(events)
            accum["run_ids"].append(np.repeat(run_id, len(events)))
            accum["vertex"].append(truth["vertex_m"][index])
    return {key: np.concatenate(parts) for key, parts in accum.items()}


def grid_case(arm, seed, label, *, device, out):
    geometry = rg.load_r2_geometry("wcte")
    scaling = rs.scaling_for(geometry, gen.ROUND1_TIME_RULE.time_scale_ns)
    model, identity = make_model(arm, seed, geometry, scaling,
                                 checkpoint_candidates("wcte", arm, seed), device)
    cohorts = [rs.open_cohort(dataset_path("wcte"), f"dev_probe_{fold}") for fold in ("train","test")]
    output = out / f"{label}.json"
    rows = []
    if output.exists():
        resume = json.loads(output.read_text())
        if resume["identity"] != identity:
            raise ValueError(f"{label}: checkpoint identity changed since previous grid cell")
        rows = resume["rows"]
    completed = {(row["d1"], row["d2"]) for row in rows}
    for name_1, gain_s in D1:
        for name_2, jitter in D2:
            if (name_1, name_2) in completed:
                continue
            train, test = [_grid_features(model, geometry, scaling, c, gain_s, jitter, device=device)
                           for c in cohorts]
            if np.intersect1d(train["run_ids"], test["run_ids"]).size:
                raise ValueError("training and test runs overlap")
            tr_r2, te_r2 = [np.concatenate([d["base"], d["contrast"]], axis=1) for d in (train, test)]
            u_train, u_test = [np.zeros((len(d["base"]), geometry.n_modules)) for d in (train, test)]
            nil_train, nil_test = u_train[:, :0], u_test[:, :0]
            target, runs = train["vertex"], train["run_ids"]
            r2 = fit_block_ridge(tr_r2, nil_train, target, runs, gammas=(np.inf,))
            split = fit_block_ridge(train["base"], u_train, target, runs)
            full = fit_block_ridge(tr_r2, u_train, target, runs)
            test_vertex = test["vertex"]
            r2_m = _mean_3d(r2.predict(te_r2, nil_test), test_vertex)
            split_m = _mean_3d(split.predict(test["base"], u_test), test_vertex)
            full_m = _mean_3d(full.predict(te_r2, u_test), test_vertex)
            row = {"model": label, "d1": name_1, "d2": name_2, "n_test": len(test_vertex),
                   "v_R2_m": r2_m, "v_R3u_split_m": split_m, "v_R3u_full_ceiling_m": full_m,
                   "split": split.settings(), "full": full.settings(),
                   "violation_split": bool(r2_m < split_m - TOLERANCE_M),
                   "violation_full": bool(r2_m < full_m - TOLERANCE_M)}
            rows.append(row)
            write_json(output, {"identity": identity, "rows": rows})
            print(json.dumps(row), flush=True)


def aggregate(out):
    original = json.loads((out / "original_s4.json").read_text())
    cells = []
    for _, _, label in CASES:
        rows = json.loads((out / f"{label}.json").read_text())["rows"]
        expected = {(d1, d2) for d1, _ in D1 for d2, _ in D2}
        if len(rows) != len(expected) or {(r["d1"], r["d2"]) for r in rows} != expected:
            raise ValueError(f"{label}: incomplete repair grid")
        cells.extend(rows)
    assert len(cells) == 54
    readings = {"extension": EXTENSION, "registered": False, "tolerance_m": TOLERANCE_M,
                "n_cells": len(cells), "n_split_violations": sum(r["violation_split"] for r in cells),
                "n_full_violations": sum(r["violation_full"] for r in cells),
                "original_s4_n_violations": sum(r["s4_violation"] for r in original),
                "s4_resolved": all(not r["s4_violation"] for r in original),
                "interpretation": {
                    "d1": "F1 gain-spread overlays on charge relative to the stored 0.05-spread WCTE runs",
                    "d2": "Event-shared Gaussian clock shifts at 25 or 40 ns, keyed by run and event",
                    "dead_module_block": "105 known-dead-module columns. Zero in the separate gain/time grid.",
                    "r2_grid": "Grouped Ridge on context z_c and complementary-view difference z_a-z_b",
                    "r2_original": "Historical frozen nominal readout after fitted nuisance correction",
                }, "grid": cells, "original_conditions": original}
    write_json(out / "p2a_repaired_readings.json", readings)
    lines = ["# WCTE P2a split-penalty Ridge", "",
             f"The 54-cell gain/time grid has {readings['n_split_violations']} split-ceiling "
             f"violations and {readings['n_full_violations']} full-ceiling violations at 5 mm.",
             f"The historical paired S4 audit has {readings['original_s4_n_violations']} residual violations.",
             "", "| Model | D1 | D2 | R2 (m) | R3u split (m) | R3u full (m) |", "|---|---|---|---:|---:|---:|"]
    lines.extend(f"| {r['model']} | {r['d1']} | {r['d2']} | {r['v_R2_m']:.5f} "
                 f"| {r['v_R3u_split_m']:.5f} | {r['v_R3u_full_ceiling_m']:.5f} |" for r in cells)
    lines.extend(["", "The historical S4 audit uses the original threshold, walk and dead-PMT conditions. "
                  "It preserves the original frozen R2 predictions and uses run-grouped fitting on the "
                  "same probe train and test partitions.", ""])
    (out / "WCTE_P2A_REPAIR.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("original", "grid", "aggregate"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=ROOT / "wcte_p2a")
    parser.add_argument("--seed", type=int, choices=(9000, 9001))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = safe_output(args.out)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.45, args.device)
    if args.mode == "original":
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = [row for block in pool.map(audit_original, [label for _, _, label in CASES])
                    for row in block]
        write_json(out / "original_s4.json", rows)
    elif args.mode == "grid":
        for arm, seed, label in CASES:
            if args.seed is None or seed == args.seed:
                grid_case(arm, seed, label, device=args.device, out=out)
    else:
        aggregate(out)


if __name__ == "__main__":
    main()
