#!/usr/bin/env python3
"""Unregistered CPU-only M-dose experiment (runbook v2, 30 Oct 2026).

Modes
-----
toy      exact-population toy curve (9 g levels, channel split, k=3)
level    one LUCiD dose level: photon event set (1000 x 10) and 1000
         calibration events x (8 scheme-A + 8 scheme-B realisations); writes
         one .npz (S, N_A, N_B, null, per-run means/gains, clip counts) and one
         compact JSON summary under the extension root
report   aggregate toy + level summaries, compute the seven pre-declared
         readings, write m_dose.md / nu_vs_r.png / JSON summaries

Everything is an extension: ``registered: false``.  No model is trained,
loaded or run; the only model path touched is the registered tokeniser.
Run from the repository root as::

    PYTHONPATH=src python -m scripts.jepa_ext_m_dose level --law F1 --spread 0.05
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from jepa_noise import lucid_bombs as lb
from jepa_noise import point_encoder as pe
from jepa_noise import views
from jepa_noise.ext_m_dose import f1x_state, m_terms, pair_diagnostics, project
from jepa_noise.ext_predictability import resample_noise
from jepa_noise.lucid_views import ViewMasker, module_hilbert_order

ROOT = Path("/ceph/dwong/jepa/returns/ext_m_dose_20261030")
REPO = Path(__file__).resolve().parents[1]
SEED = 20261017
DEFAULT_CONFIG = "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json"
DEFAULT_GEOMETRY = "/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json"
N_RUNS, N_EVENTS, N_CAL, REPLICATES = 1000, 10, 1000, 8
TOY_LEVELS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.225, 0.45, 0.9)
LEVELS = (("F1", 0.0), ("F1", 0.02), ("F1", 0.05), ("F1", 0.10), ("F1", 0.20),
          ("F1", 0.30), ("F1x", 0.50), ("F1x", 0.75), ("F1x", 1.00), ("F1x", 1.50))
CRITERION_K = 4


def law_label(law):
    return "F1" if law == "F1" else "F1x_stress_unphysical"


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
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


def check_provenance(config_path):
    """Repo config must equal the sha256 recorded by the pre-grid provenance."""
    path = Path(config_path)
    digest = sha256_file(path)
    recorded = json.loads(
        (REPO / "results/jepa/20261025_pregrid_checks/provenance.json").read_text())
    expected = recorded["generator_config"]["sha256"]
    if digest != expected:
        raise SystemExit(f"config hash mismatch: {digest} != {expected}")
    return {"config_path": str(path.resolve()), "sha256": digest,
            "provenance_recorded_sha256": expected,
            "provenance": "results/jepa/20261025_pregrid_checks/provenance.json"}


def load_inputs(args):
    provenance = check_provenance(args.config)
    geometry = lb.load_wcte_geometry(args.geometry)
    cfg = lb.load_generator_config(args.config)
    provenance["geometry"] = {
        "path": str(Path(args.geometry).resolve()),
        "sha256": sha256_file(args.geometry),
        "config_sha256": geometry.config_sha256,
        "source_sha256": geometry.source_sha256,
        "digest": geometry.digest,
        "n_pmt": int(geometry.n_pmt),
        "n_modules": int(geometry.n_modules),
    }
    return geometry, cfg, provenance


def state_for(geometry, law, spread, run_id, *, seed=SEED):
    if law == "F1":
        return lb.state_for_run("F1", float(spread), run_id, geometry, seed=seed)
    if law == "F1x":
        return f1x_state(geometry, float(spread), run_id, seed=seed)
    raise ValueError(f"unknown state law {law!r}")


def view_groups(geometry):
    """The 29 Oct Part-2 module-pooled view groups."""
    order = module_hilbert_order(geometry)
    middle = len(order) // 2
    module = geometry.module_of_pmt
    parity = geometry.pmt_id_in_mpmt % 2
    return {
        "within": ([np.flatnonzero((module == m) & (parity == 0))
                    for m in range(geometry.n_modules)],
                   [np.flatnonzero((module == m) & (parity == 1))
                    for m in range(geometry.n_modules)]),
        "across_fixed": ([np.flatnonzero(module == m) for m in order[:middle]],
                         [np.flatnonzero(module == m) for m in order[middle:]]),
    }


def pool(features, groups):
    """Per module/view mean of (log charge, time, hit flag), 29 Oct Part 2."""
    return np.stack([features[:, ids, 6:9].mean(axis=1) for ids in groups],
                    axis=1).reshape(len(features), -1)


def tokenise_hits(geometry, hits, scaling, masker):
    """Tokenise an in-memory batch; raw events are never written to disk."""
    store = lb.sparse_from_readouts(np.arange(len(hits), dtype=np.int64), hits)
    original = pe.read_sparse_hits
    pe.read_sparse_hits = lambda path: store
    try:
        batch = pe.tokenise("in_memory", geometry, split="within", seed=0,
                            scaling=scaling, masker=masker)
    finally:
        pe.read_sparse_hits = original
    if len(batch.event_ids) != len(hits):
        raise RuntimeError("tokeniser unexpectedly excluded an event at n_min=0")
    return (batch.features.numpy().astype(np.float64),
            batch.target_index[0].numpy())


def below_n_min(features, target_index):
    hits = (features[:, target_index, 8] > 0).sum(axis=1)
    return int(np.sum(hits < lb.N_MIN))


def collect(geometry, cfg, law, spread):
    """Stream one dose level into float64 running sums (runbook §3.2)."""
    readout = lb.make_readout(geometry, cfg)
    scaling = pe.FeatureScaling.from_geometry(geometry)
    masker = ViewMasker(geometry, "within", n_min=0)
    groups = view_groups(geometry)
    dims = {name: 3 * (len(groups[name][0]) + len(groups[name][1])) for name in groups}
    acc = {name: {"sum": np.zeros(dim), "outer": np.zeros((dim, dim)),
                  "means": np.empty((N_RUNS, dim)),
                  "A": np.zeros((dim, dim)), "B": np.zeros((dim, dim)),
                  "null": np.zeros((dim, dim))}
           for name, dim in dims.items()}
    gain = np.empty((N_RUNS, geometry.n_modules))
    clipped_modules = 0
    below_events, below_cal = 0, 0
    start = time.monotonic()

    # Event set: 1000 runs x 10 events, common random numbers across levels.
    for run in range(N_RUNS):
        run_id = f"ext_m_dose/evt/{run:04d}"
        state = state_for(geometry, law, spread, run_id)
        gain[run] = np.log(state.module_gain)
        clipped_modules += int(state.clipped["module"])
        hits = [readout(state, lb.draw_event(geometry, cfg, run_id,
                                             run * N_EVENTS + event,
                                             seed=SEED, domain="photon"))
                for event in range(N_EVENTS)]
        features, target_index = tokenise_hits(geometry, hits, scaling, masker)
        below_events += below_n_min(features, target_index)
        for name in groups:
            x = pool(features, groups[name])
            acc[name]["means"][run] = x.mean(axis=0)
            acc[name]["sum"] += x.sum(axis=0)
            acc[name]["outer"] += x.T @ x
        if run % 100 == 0:
            print(f"event set {law} {spread}: {run}/{N_RUNS}", flush=True)
    n_event_vectors = N_RUNS * N_EVENTS
    for name in groups:
        mean = acc[name]["sum"] / n_event_vectors
        acc[name]["S"] = ((acc[name]["outer"]
                           - n_event_vectors * np.outer(mean, mean))
                          / (n_event_vectors - 1))
        del acc[name]["outer"], acc[name]["sum"]

    # Calibration set: 1000 events, 8 scheme-A and 8 scheme-B realisations.
    previous = {name: None for name in groups}
    first_target = {name: None for name in groups}
    scale = np.sqrt(REPLICATES / (REPLICATES - 1))
    for event in range(N_CAL):
        run_id = f"ext_m_dose/cal/{event:04d}"
        draws = lb.draw_event(geometry, cfg, run_id, event, seed=SEED,
                              domain="photon")
        fixed_state = state_for(geometry, law, spread, run_id)
        hits_a, hits_b = [], []
        for replicate in range(REPLICATES):
            hits_a.append(readout(fixed_state, resample_noise(
                draws, geometry, cfg, replicate, seed=SEED)))
            fresh_seed = int(lb.keyed_rng(
                SEED, "ext_predictability_state", run_id, event, replicate)
                .integers(0, 2**64, dtype=np.uint64))
            fresh_state = state_for(geometry, law, spread, run_id, seed=fresh_seed)
            hits_b.append(readout(fresh_state, resample_noise(
                draws, geometry, cfg, replicate + 1, seed=SEED)))
        features_a, target_index = tokenise_hits(geometry, hits_a, scaling, masker)
        features_b, _ = tokenise_hits(geometry, hits_b, scaling, masker)
        below_cal += below_n_min(features_a, target_index)
        below_cal += below_n_min(features_b, target_index)
        for name in groups:
            context_dim = 3 * len(groups[name][0])
            xa, xb = pool(features_a, groups[name]), pool(features_b, groups[name])
            ra = (xa - xa.mean(axis=0)) * scale
            rb = (xb - xb.mean(axis=0)) * scale
            acc[name]["A"] += ra.T @ ra / REPLICATES
            acc[name]["B"] += rb.T @ rb / REPLICATES
            if previous[name] is not None:
                acc[name]["null"][:context_dim, context_dim:] += (
                    previous[name].T @ rb[:, context_dim:] / REPLICATES)
            else:
                first_target[name] = rb[:, context_dim:].copy()
            previous[name] = rb[:, :context_dim].copy()
        if event % 50 == 0:
            print(f"calibration {law} {spread}: {event}/{N_CAL}", flush=True)
    for name in groups:
        context_dim = 3 * len(groups[name][0])
        acc[name]["null"][:context_dim, context_dim:] += (
            previous[name].T @ first_target[name] / REPLICATES)
        acc[name]["A"] /= N_CAL
        acc[name]["B"] /= N_CAL
        acc[name]["null"] /= N_CAL

    arrays = {
        "gain": gain,
        "clipped_modules": np.asarray(clipped_modules),
        "clipped_fraction": np.asarray(
            clipped_modules / (N_RUNS * geometry.n_modules)),
        "below_event_set": np.asarray(below_events),
        "below_calibration": np.asarray(below_cal),
        "n_runs": np.asarray(N_RUNS), "n_events_per_run": np.asarray(N_EVENTS),
        "n_calibration": np.asarray(N_CAL), "replicates": np.asarray(REPLICATES),
        "seed": np.asarray(SEED), "law": np.asarray(law),
        "setting": np.asarray(float(spread)),
        "wall_seconds": np.asarray(time.monotonic() - start),
    }
    for name, dim in dims.items():
        arrays[f"{name}_dim"] = np.asarray(dim)
        arrays[f"{name}_context_dim"] = np.asarray(3 * len(groups[name][0]))
        arrays[f"{name}_target_dim"] = np.asarray(3 * len(groups[name][1]))
        for key in ("S", "A", "B", "null", "means"):
            arrays[f"{name}_{key}"] = acc[name][key]
    return arrays


def analyse(data, law, spread):
    """Per-level covariance analysis (runbook §2.2-§2.4, §3.2)."""
    from scripts.jepa_pregrid_checks import ridge_r2

    rows = []
    gain = data["gain"]
    for name in ("within", "across_fixed"):
        S, A, B, null = (data[f"{name}_S"], data[f"{name}_A"],
                         data[f"{name}_B"], data[f"{name}_null"])
        context_dim = int(data[f"{name}_context_dim"])
        target_dim = int(data[f"{name}_target_dim"])
        if S.shape != (context_dim + target_dim,) * 2:
            raise ValueError(f"{name}: stored covariance shape {S.shape} does "
                             f"not match {context_dim}+{target_dim}")
        blocks = (S[:context_dim, :context_dim], S[context_dim:, context_dim:],
                  S[:context_dim, context_dim:],
                  B[:context_dim, :context_dim], B[context_dim:, context_dim:],
                  B[:context_dim, context_dim:])
        diagnostics = pair_diagnostics(*blocks, CRITERION_K,
                                       A[:context_dim, context_dim:],
                                       (B - A)[:context_dim, context_dim:],
                                       null[:context_dim, context_dim:])
        canonical = diagnostics.pop("canonical")
        context = diagnostics.pop("context")
        diagnostics.pop("target")
        shares = diagnostics.pop("shares")
        registered_blocks = project(blocks, name)
        m_reg = m_terms(*registered_blocks, CRITERION_K)[0]
        sigma_null = m_terms(blocks[0], blocks[1], blocks[2], blocks[3],
                             blocks[4], null[:context_dim, context_dim:],
                             CRITERION_K)[2][0]
        mean_projection = data[f"{name}_means"][:, :context_dim] @ context
        r2_gain = [ridge_r2(gain[:N_RUNS // 2], mean_projection[:N_RUNS // 2, j],
                            gain[N_RUNS // 2:], mean_projection[N_RUNS // 2:, j])["r2"]
                   for j in range(CRITERION_K)]
        row = {
            "testbed": "LUCiD", "law": law_label(law), "law_key": law,
            "level": float(spread), "split": name,
            "M_reg": m_reg, "M_full": diagnostics["M_full"],
            "sigma_phys": diagnostics["sigma_phys"],
            "sigma_noise": diagnostics["sigma_noise"],
            "sigma_noise_null": float(sigma_null),
            "r": diagnostics["ratio"], "r_s": diagnostics["ratio_s"],
            "r_null": (float(sigma_null / diagnostics["sigma_phys"])
                       if diagnostics["sigma_phys"] else None),
            "at_floor": bool(diagnostics["sigma_noise"] <= 1.2 * sigma_null),
            "nu": shares[0],
            "nu_A": shares[1], "nu_gain": shares[2], "nu_null": shares[3],
            "nu_star": float(np.max(shares[0])),
            "nu_star_A": float(np.max(shares[1])),
            "nu_star_gain": float(np.max(shares[2])),
            "nu_star_null": float(np.max(shares[3])),
            "nu_k": diagnostics["aggregate"],
            "r2_gain": r2_gain,
            "canonical": canonical,
            "identity_error": diagnostics["identity_error"],
            "out_of_unit_interval": int(np.sum((shares[0] < 0) | (shares[0] > 1))),
            "clipped_fraction": float(data["clipped_fraction"]),
            "below_event_set": int(data["below_event_set"]),
            "below_calibration": int(data["below_calibration"]),
            "wall_seconds": float(data["wall_seconds"]),
        }
        rows.append(jsonable(row))
    return rows


def toy():
    """Exact-population toy curve: A4 linearised fixture, base rho=0, E2 theta."""
    rows = []
    for g in TOY_LEVELS:
        fixture = views.Tier1Views(split="channel", line_multiplier=0.,
                                   common_strength=g)
        population = fixture.population()
        blocks = tuple(population[key] for key in
                       ("scc", "stt", "sct", "ncc", "ntt", "nct"))
        diagnostics = pair_diagnostics(*blocks, 3)
        canonical = diagnostics.pop("canonical")
        diagnostics.pop("context")
        diagnostics.pop("target")
        shares = diagnostics.pop("shares")
        row = {
            "testbed": "toy", "law": "toy", "law_key": "toy",
            "level": float(g), "split": "channel",
            "M_reg": m_terms(*project(blocks, "channel"), 3)[0],
            "M_full": diagnostics["M_full"],
            "sigma_phys": diagnostics["sigma_phys"],
            "sigma_noise": diagnostics["sigma_noise"],
            "sigma_noise_null": None, "r": diagnostics["ratio"],
            "r_s": diagnostics["ratio_s"], "r_null": None, "at_floor": None,
            "nu": shares[0], "nu_A": None, "nu_gain": None, "nu_null": None,
            "nu_star": float(np.max(shares[0])), "nu_star_A": None,
            "nu_star_gain": None, "nu_star_null": None,
            "nu_k": diagnostics["aggregate"], "r2_gain": None,
            "canonical": canonical,
            "identity_error": diagnostics["identity_error"],
            "out_of_unit_interval": int(np.sum((shares[0] < 0) | (shares[0] > 1))),
            "clipped_fraction": 0.0, "below_event_set": None,
            "below_calibration": None, "wall_seconds": None,
            "D": float(fixture.displacement(3)["angle_deg"]),
        }
        rows.append(jsonable(row))
    return rows


def spearman(x, y):
    from scipy.stats import spearmanr
    values = spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return None if not np.isfinite(values.statistic) else float(values.statistic)


def readings(toy_rows, level_rows):
    """The seven pre-declared descriptive readings (runbook §4)."""
    within = [r for r in level_rows if r["split"] == "within"]
    across = [r for r in level_rows if r["split"] == "across_fixed"]
    nonzero = [r for r in within if r["level"] > 0]
    result = {}

    m1 = spearman([r["r"] for r in nonzero], [r["nu_star"] for r in nonzero])
    result["M-1"] = {
        "value": m1, "n_levels": len(nonzero),
        "supported": ("supported" if m1 is not None and m1 >= 0.8 else
                      "not supported" if m1 is not None else "unclear"),
        "toy_curve": [[r["r"], r["nu_star"]] for r in toy_rows],
    }

    checks = []
    for level in (0.05, 0.10):
        row = next((r for r in within
                    if r["law_key"] == "F1" and abs(r["level"] - level) < 1e-12),
                   None)
        if row is None:
            checks.append({"level": level, "present": False})
            continue
        checks.append({
            "level": level, "present": True,
            "r": row["r"], "r_below_0.530": bool(row["r"] is not None
                                               and row["r"] < 0.530),
            "at_floor": row["at_floor"], "nu_star": row["nu_star"],
            "nu_star_gain": row["nu_star_gain"],
        })
    m2_ok = all(c.get("present") and
                ((c["r_below_0.530"] or c["at_floor"]) and c["nu_star"] <= 0.1)
                for c in checks)
    result["M-2"] = {"levels": checks,
                     "supported": "supported" if checks and m2_ok else
                     "not supported" if checks else "not_evaluable"}

    trap = [r for r in within if r["nu_star"] >= 0.5
            and r["nu_star"] - r["nu_star_null"] >= 0.2]
    result["M-3"] = {
        "levels": [[r["law_key"], r["level"], r["nu_star"],
                    r["nu_star_null"]] for r in within],
        "supported": "supported" if trap else "not supported",
    }

    if trap:
        first = sorted(trap, key=lambda r: r["level"])[0]
        onset = {
            "law": first["law_key"], "level": first["level"],
            "r": first["r"], "r_s": first["r_s"],
            "in_window": bool(first["r"] is not None
                              and 0.6 <= first["r"] <= 1.2),
            "toy_onset_r_3": 1.036,
        }
        result["M-4"] = {
            "onset": onset,
            "supported": "supported" if onset["in_window"] else "not supported",
        }
    else:
        result["M-4"] = {"onset": None, "supported": "not_evaluable"}

    high = [r for r in within if r["nu_star_gain"] >= 0.5]
    low = [r for r in within if r["nu_star_gain"] <= 0.1]
    def _best(row):
        values = [value for value in row["r2_gain"] if value is not None]
        return max(values) if values else None

    high_ok = all(_best(r) is not None and _best(r) >= 0.5 for r in high)
    low_ok = all(_best(r) is None or _best(r) < 0.5 for r in low)
    if high and low:
        m5 = "supported" if (high_ok and low_ok) else "not supported"
    else:
        m5 = "not_evaluable"
    result["M-5"] = {
        "high_levels": [[r["law_key"], r["level"], r["nu_star_gain"],
                         _best(r)] for r in high],
        "low_levels": [[r["law_key"], r["level"], r["nu_star_gain"],
                        _best(r)] for r in low],
        "supported": m5,
    }

    graded = [r for r in within if r["nu_star"] >= 0.2]
    checks_m6 = []
    for row in graded:
        other = next((r for r in across
                      if r["law_key"] == row["law_key"]
                      and abs(r["level"] - row["level"]) < 1e-12), None)
        if other is not None:
            checks_m6.append((row["law_key"], row["level"],
                              row["nu_star_gain"], other["nu_star_gain"]))
    m6_ok = bool(checks_m6) and all(a < b for _, _, a, b in checks_m6)
    result["M-6"] = {
        "levels": checks_m6,
        "supported": ("supported" if m6_ok else
                      "not supported" if checks_m6 else "not_evaluable"),
    }

    toy_rho = spearman([r["M_reg"] for r in toy_rows],
                       [r["M_full"] for r in toy_rows])
    lucid_rho = spearman([r["M_reg"] for r in within],
                         [r["M_full"] for r in within])
    lucid_all = spearman([r["M_reg"] for r in level_rows],
                         [r["M_full"] for r in level_rows])
    m7_ok = (toy_rho is not None and lucid_rho is not None
             and toy_rho >= 0.8 and lucid_rho >= 0.8)
    result["M-7"] = {
        "toy": toy_rho, "lucid_within": lucid_rho,
        "lucid_all_splits": lucid_all,
        "supported": "supported" if m7_ok else "not supported",
    }
    return result


def markdown_table(toy_rows, level_rows):
    header = ("| testbed | law | level | split | M_reg | M_full | sigma_k_phys | "
              "sigma_1_noise | r_k | r_k^S | nu* (N_B) | nu* (N_A) | nu* (N_gain) | "
              "nu*_null | nu_k | R2_gain top4 / D | clipped frac |")
    rule = "|" + "---|" * 17
    lines = [header, rule]
    for row in toy_rows + level_rows:
        r2 = ("D=%s" % _num(row["D"]) if row["testbed"] == "toy" else
              ", ".join(_num(value) for value in row["r2_gain"]))
        lines.append("| " + " | ".join([
            row["testbed"], row["law"], _num(row["level"]), row["split"],
            _num(row["M_reg"]), _num(row["M_full"]), _num(row["sigma_phys"]),
            _num(row["sigma_noise"]), _num(row["r"]), _num(row["r_s"]),
            _num(row["nu_star"]), _num(row.get("nu_star_A"), "—"),
            _num(row.get("nu_star_gain"), "—"),
            _num(row.get("nu_star_null"), "—"), _num(row["nu_k"]), r2,
            _num(row["clipped_fraction"]),
        ]) + " |")
    return "\n".join(lines) + "\n"


def _num(value, missing="—", digits=4):
    if value is None:
        return missing
    return f"{float(value):.{digits}f}"


def plot(toy_rows, level_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7.0, 5.0), dpi=150)
    floors = [r["nu_star_null"] for r in level_rows
              if r.get("nu_star_null") is not None]
    if floors:
        axis.axhspan(0.0, max(floors) + 0.05, color="0.85", zorder=0,
                     label="shuffled-null floor band")
    axis.plot([r["r"] for r in toy_rows], [r["nu_star"] for r in toy_rows],
              "o-", color="tab:blue", label="toy (population)")
    for law, colour, marker in (("F1", "tab:orange", "s"),
                                ("F1x_stress_unphysical", "tab:red", "^")):
        rows = [r for r in level_rows if r["law"] == law and r["split"] == "within"]
        rows = [r for r in rows if r["r"] is not None]
        axis.plot([r["r"] for r in rows], [r["nu_star"] for r in rows],
                  marker + ":", color=colour, label=f"LUCiD {law}")
        floor_rows = [r for r in rows if r["at_floor"]]
        axis.plot([r["r"] for r in floor_rows],
                  [r["nu_star"] for r in floor_rows],
                  marker, markerfacecolor="none", markersize=11,
                  markeredgecolor=colour)
    axis.axvline(1.0, color="0.4", linestyle="--", linewidth=1)
    axis.set_xlabel("r_k = sigma_1^noise / sigma_k^phys")
    axis.set_ylabel("nu* = max_j<=k nu_(j)  (N_B)")
    axis.set_title("M dose: nuisance share vs the ratio of M's two terms")
    axis.set_ylim(bottom=-0.02)
    axis.legend(fontsize=8, loc="upper left")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def run_toy(args):
    rows = toy()
    payload = {"registered": False, "extension": "m_dose_2026-10-30",
               "kind": "toy", "rows": rows,
               "acceptance": {
                   "g": [0.0, 0.005, 0.02, 0.05],
                   "nu_3": [0.0, 0.043, 0.525, 0.918],
                   "r_3": [0.0, 0.530, 1.036, 1.281],
                   "D": [0.0, 11.27, 47.96, 76.95],
                   "M_full": [0.682, 0.317, -0.024, -0.186],
                   "M_reg": [0.0099, -0.0025, -0.038, -0.102],
               }}
    write_json(Path(args.out_root) / "toy.json", payload)
    for row in rows:
        print(f"toy g={row['level']:<6} nu*={row['nu_star']:.4f} r={row['r']:.4f} "
              f"D={row['D']:.2f} M_full={row['M_full']:+.4f} M_reg={row['M_reg']:+.4f}")


def run_level(args):
    if (args.law, float(args.spread)) not in LEVELS:
        raise SystemExit(f"unknown level {(args.law, args.spread)}")
    geometry, cfg, provenance = load_inputs(args)
    root = Path(args.out_root)
    if not root.is_dir():
        raise SystemExit(f"extension root must exist: {root}")
    stem = f"{args.law}_{float(args.spread):.2f}"
    npz_path = root / f"{stem}.npz"
    json_path = root / f"{stem}.json"
    if npz_path.exists() or json_path.exists():
        raise SystemExit(f"refusing to overwrite {npz_path} / {json_path}")
    start = time.monotonic()
    data = collect(geometry, cfg, args.law, float(args.spread))
    with npz_path.open("xb") as stream:
        np.savez_compressed(stream, **data)
    rows = analyse(data, args.law, float(args.spread))
    smoke = None
    if args.law == "F1" and abs(float(args.spread) - 0.05) < 1e-12:
        within = next(row for row in rows if row["split"] == "within")
        smoke = {
            "runs_end_to_end": True,
            "top4_within_canonical": within["canonical"],
            "canonical_all_gt_0.8": bool(all(c > 0.8
                                            for c in within["canonical"])),
            "r2_gain_top4": within["r2_gain"],
            "r2_gain_all_lt_0.1": bool(all(v is not None and v < 0.1
                                           for v in within["r2_gain"])),
        }
    payload = {
        "registered": False, "extension": "m_dose_2026-10-30", "kind": "level",
        "law": args.law, "law_label": law_label(args.law),
        "setting": float(args.spread), "rows": rows, "smoke": smoke,
        "provenance": provenance,
        "npz_sha256": sha256_file(npz_path),
        "npz_bytes": npz_path.stat().st_size,
        "wall_seconds": time.monotonic() - start,
        "n_runs": N_RUNS, "n_events_per_run": N_EVENTS,
        "n_calibration": N_CAL, "replicates": REPLICATES,
    }
    write_json(json_path, payload)
    for row in rows:
        print(f"{args.law} {row['level']} {row['split']}: M={row['M_full']:+.4f} "
              f"r={row['r']} nu*={row['nu_star']:.4f} r2={row['r2_gain']}")


def run_report(args):
    root = Path(args.out_root)
    results = Path(args.results_dir)
    toy_payload = json.loads((root / "toy.json").read_text())
    toy_rows = toy_payload["rows"]
    level_payloads = []
    for law, spread in LEVELS:
        path = root / f"{law}_{spread:.2f}.json"
        if path.exists():
            level_payloads.append(json.loads(path.read_text()))
    level_rows = [row for payload in level_payloads for row in payload["rows"]]
    readings_payload = readings(toy_rows, level_rows)
    npz_hashes = {}
    for payload in level_payloads:
        stem = f"{payload['law']}_{payload['setting']:.2f}.npz"
        npz_hashes[stem] = {
            "path": str(root / stem), "sha256": payload["npz_sha256"],
            "bytes": payload["npz_bytes"],
        }
    measured = {f"{payload['law']}_{payload['setting']:.2f}":
                payload["wall_seconds"] for payload in level_payloads}
    summary = {
        "registered": False, "extension": "m_dose_2026-10-30",
        "levels_present": [payload["law"] + "_" + f"{payload['setting']:.2f}"
                           for payload in level_payloads],
        "levels_missing": [law + "_" + f"{spread:.2f}" for law, spread in LEVELS
                           if not (root / f"{law}_{spread:.2f}.json").exists()],
        "measured_wall_seconds": measured,
        "measured_wall_hours_total": sum(measured.values()) / 3600.0,
        "smoke": next((payload["smoke"] for payload in level_payloads
                       if payload["smoke"] is not None), None),
        "provenance": level_payloads[0]["provenance"] if level_payloads else None,
    }
    results.mkdir(parents=True, exist_ok=True)
    write_json(results / "readings.json", readings_payload, overwrite=True)
    write_json(results / "summary.json", summary, overwrite=True)
    write_json(results / "npz_hashes.json", npz_hashes, overwrite=True)
    write_json(results / "toy.json", toy_payload, overwrite=True)
    for payload in level_payloads:
        write_json(results / f"{payload['law']}_{payload['setting']:.2f}.json",
                   payload, overwrite=True)
    (results / "m_dose.md").write_text(
        "# M dose — one table (extension, registered: false)\n\n"
        + markdown_table(toy_rows, level_rows))
    plot(toy_rows, level_rows, results / "nu_vs_r.png")
    lines = ["# M dose extension (30 Oct 2026)", "",
             "**registered: false** — extension `m_dose_2026-10-30`.", "",
             "Runbook: `docs/jepa/RUNBOOK_2026-10-30_M_DOSE.md` (v2).",
             "No model is trained, loaded or run; CPU only.", "",
             "Contents: `m_dose.md` (table), `nu_vs_r.png`, `readings.json`, "
             "`summary.json`, `npz_hashes.json`, `toy.json`, per-level JSON.", "",
             "The `.npz` covariance files stay under "
             f"`{root}`; their hashes are in `npz_hashes.json`.", ""]
    (results / "README.md").write_text("\n".join(lines))
    print(json.dumps(jsonable(readings_payload), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("toy", "level", "report"))
    parser.add_argument("--out-root", default=str(ROOT))
    parser.add_argument("--results-dir",
                        default=str(REPO / "results/jepa/20261030_m_dose"))
    parser.add_argument("--law", choices=("F1", "F1x"))
    parser.add_argument("--spread", type=float)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--geometry", default=DEFAULT_GEOMETRY)
    args = parser.parse_args(argv)
    if args.mode == "toy":
        run_toy(args)
    elif args.mode == "level":
        if args.law is None or args.spread is None:
            raise SystemExit("level mode needs --law and --spread")
        run_level(args)
    else:
        run_report(args)


if __name__ == "__main__":
    main()
