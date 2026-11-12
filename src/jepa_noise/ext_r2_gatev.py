"""Round-2 Gate V photon-statistics ceiling (addendum RUNBOOK_2026-11-12_R2_GATEV_CEILING.md §2).

A diagnostic only: it never gates anything.  For each detector and holdout event it asks what
a *perfect* analytic model would score against a finite-photon reference: the Λ incident photons
of the committed Gate V row are redistributed as multinomial(Λ, μ / Σμ) with detected charge
n* · w̄, and each draw is scored with the harness arithmetic of
``cherenkov_validation.compare_event`` (same ``keep`` rule, ``log1p``, module pooling).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import lucid_bombs as lb
from . import r2_geometry as rg

ADDENDUM_SHA256 = "d237ec2b3059f5359b51e20e2047b78454264b214cd73c8699d7d7030cca582c"
CEILING_SEED = 20261112
K_DRAWS = 500
MU_REL_TOL = 1e-6
METRICS = ("pattern_pearson", "log_pattern_pearson", "module_pattern_pearson")
SPLIT_HALF_DRAWS = 100


# ------------------------------------------------------------------ the compare_event arithmetic
def expected_mu(geometry, cfg, event) -> np.ndarray:
    """μ per PMT exactly as ``cherenkov_validation.compare_event`` computes it."""
    from . import cherenkov_validation as cv  # noqa: F401  (same module family, same lb calls)
    if event.domain == lb.DOMAIN_PHOTON:
        qe = np.full(geometry.n_pmt, float(lb.qe_at(np.array([450.0]), cfg)[0]))
        return lb.expected_charge(geometry, event.vertex_m, event.yield_photons, qe,
                                  lb.LAMBDA0_M, cfg)
    truth = lb.ParticleTruth(domain=event.domain, vertex_m=event.vertex_m,
                             direction=event.direction_m, energy_MeV=event.energy_MeV,
                             deposited_MeV=event.deposited_MeV, contained=event.contained,
                             track_length_m=event.track_length_m)
    qe_mean = float(np.mean(lb.qe_at(event.wavelength_nm, cfg))) if event.wavelength_nm.size else 0.2
    mu, _ = lb.expected_charge_particle(geometry, truth, np.full(geometry.n_pmt, qe_mean),
                                        lb.LAMBDA0_M, cfg)
    return np.asarray(mu, float)


def score(mu: np.ndarray, counts: np.ndarray, detected: np.ndarray, module: np.ndarray,
          n_modules: int) -> dict:
    """The three pattern metrics of ``compare_event`` for one reference (counts, detected)."""
    keep = (counts > 0) & (mu > 0)
    nan = float("nan")
    pattern = float(np.corrcoef(mu[keep], detected[keep])[0, 1]) if keep.sum() >= 3 else nan
    log_pattern = (float(np.corrcoef(np.log1p(mu[keep]), np.log1p(detected[keep]))[0, 1])
                   if keep.sum() >= 3 else nan)
    observed_module = np.bincount(module, weights=detected, minlength=n_modules)
    model_module = np.bincount(module, weights=mu, minlength=n_modules)
    module_pattern = float(np.corrcoef(model_module, observed_module)[0, 1])
    return {"pattern_pearson": pattern, "log_pattern_pearson": log_pattern,
            "module_pattern_pearson": module_pattern}


def perfect_draw(mu: np.ndarray, n_photons: int, w_bar: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """One perfect-model reference: Λ photons ~ multinomial(Λ, μ/Σμ), detected = n* · w̄."""
    p = np.asarray(mu, float) / float(np.sum(mu))
    n_star = rng.multinomial(int(n_photons), p).astype(float)
    return n_star, n_star * float(w_bar)


def event_ceiling(mu, n_photons: int, w_bar: float, module, n_modules: int, *, detector: str,
                  event: int, k: int = K_DRAWS, seed: int = CEILING_SEED) -> dict:
    module = np.asarray(module, np.int64)
    draws = {m: [] for m in METRICS}
    for d in range(int(k)):
        rng = lb.keyed_rng(seed, "r2/gatev_ceiling", detector, int(event), d)
        counts, detected = perfect_draw(mu, n_photons, w_bar, rng)
        s = score(mu, counts, detected, module, n_modules)
        for m in METRICS:
            draws[m].append(s[m])
    out = {}
    for m in METRICS:
        v = np.asarray(draws[m], float)
        out[m] = {"c": float(np.nanmedian(v)), "p05": float(np.nanpercentile(v, 5)),
                  "p95": float(np.nanpercentile(v, 95))}
    return out


def split_half(event, geometry, cfg, mu, *, detector: str, event_index: int,
               draws: int = SPLIT_HALF_DRAWS, seed: int = CEILING_SEED) -> dict:
    """Split-half reliability of the transported reference (Spearman–Brown, then √)."""
    n = geometry.n_pmt
    module = np.asarray(geometry.module_index, np.int64)
    w = lb.qe_at(event.wavelength_nm, cfg) * float(cfg.ce)
    res = {m: [] for m in METRICS}
    for d in range(int(draws)):
        rng = lb.keyed_rng(seed, "r2/gatev_ceiling/split", detector, int(event_index), d)
        half = rng.random(event.sensor_id.size) < 0.5
        a = [np.bincount(event.sensor_id[sel], minlength=n).astype(float) for sel in (half, ~half)]
        q = [np.bincount(event.sensor_id[sel], weights=w[sel], minlength=n) for sel in (half, ~half)]
        keep = (a[0] > 0) & (a[1] > 0) & (mu > 0)
        r = {"pattern_pearson": float(np.corrcoef(q[0][keep], q[1][keep])[0, 1]),
             "log_pattern_pearson": float(np.corrcoef(np.log1p(q[0][keep]),
                                                      np.log1p(q[1][keep]))[0, 1]),
             "module_pattern_pearson": float(np.corrcoef(
                 np.bincount(module, weights=q[0], minlength=geometry.n_modules),
                 np.bincount(module, weights=q[1], minlength=geometry.n_modules))[0, 1])}
        for m in METRICS:
            res[m].append(r[m])
    out = {}
    for m in METRICS:
        r = float(np.nanmedian(res[m]))
        sb = 2.0 * r / (1.0 + r) if r > -1 else float("nan")
        out[m] = {"split_half_r": r, "spearman_brown": sb,
                  "sqrt_reliability": float(np.sqrt(sb)) if sb >= 0 else float("nan")}
    return out


# ------------------------------------------------------------------ transport (as ext_r2.gate_v)
def transported_events(detector: str, cfg, cache_dir: Path | None = None) -> list:
    """The ten holdout events through the pinned LUCiD transport, exactly as ``ext_r2.gate_v``
    builds them; cached as npz so the diagnostic can be re-run cheaply."""
    import sys
    from . import cherenkov_validation as cv
    from . import ext_r2 as x
    geometry = rg.load_r2_geometry(detector)
    cached = []
    if cache_dir is not None:
        paths = [Path(cache_dir) / f"{detector}__{n}" for n in x.GATE_V_HOLDOUT]
        if all(p.is_file() for p in paths):
            return [cv.load_event_npz(p) for p in paths]
    pinned = str(rg.lucid_config_dir().parent)
    if pinned not in sys.path[:1]:
        sys.path.insert(0, pinned)
    from noise_module_lucid.lucid_transport import PromptTransport
    config_path = rg.lucid_config_dir() / rg.DETECTORS[detector][0]
    transport = PromptTransport(config_path, lb._repo_root() / x.UPSTREAM_MANIFEST_RELPATH)
    mapping = x.tolerant_sensor_map(geometry, transport.geometry.sensor_points)
    for index, name in enumerate(x.GATE_V_HOLDOUT):
        with np.load(Path(x.PHOTONSIM_DIR) / name, allow_pickle=False) as data:
            raw = {k: data[k] for k in data.files}
        raw = cv.contain_event(raw, geometry)
        batch = transport.export(index, np.asarray(raw["photon_id"], np.int64),
                                 np.asarray(raw["origins_m"], float),
                                 np.asarray(raw["directions"], float),
                                 np.asarray(raw["emission_ns"], float),
                                 np.asarray(raw["wavelength_nm"], float))
        raw_sensor = np.asarray(batch.sensor_id, np.int64)
        keep = mapping[raw_sensor] >= 0
        kept = SimpleNamespace(sensor_id=raw_sensor[keep],
                               arrival_time_ns=np.asarray(batch.arrival_time_ns)[keep],
                               wavelength_nm=np.asarray(batch.wavelength_nm)[keep])
        ev = cv.event_from_transport(raw, kept, event_id=index,
                                     sensor_map=np.where(mapping >= 0, mapping, 0))
        cached.append(ev)
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            fields = {k: getattr(ev, k) for k in ("event_id", "domain", "vertex_m", "sensor_id",
                                                    "time_ns", "wavelength_nm", "energy_MeV",
                                                    "deposited_MeV", "contained",
                                                    "track_length_m", "yield_photons")}
            if ev.direction_m is not None:
                fields["direction_m"] = ev.direction_m
            np.savez(Path(cache_dir) / f"{detector}__{name}", **fields)
    return cached


def detector_ceiling(detector: str, committed_rows: list, cfg, *, events=None,
                     k: int = K_DRAWS, with_split_half: bool = True) -> dict:
    geometry = rg.load_r2_geometry(detector)
    module = np.asarray(geometry.module_index, np.int64)
    rows = []
    for i, row in enumerate(committed_rows):
        ev = events[i]
        mu = expected_mu(geometry, cfg, ev)
        rel = abs(float(mu.sum()) - float(row["expected_detected"])) / float(row["expected_detected"])
        if rel > MU_REL_TOL:
            return {"detector": detector, "skipped": True,
                    "reason": f"event {i}: recomputed expected_detected differs by {rel:.3g} "
                              f"relative (> {MU_REL_TOL})"}
        lam = int(row["n_photons"])
        w_bar = float(row["observed_detected"]) / lam
        c = event_ceiling(mu, lam, w_bar, module, geometry.n_modules, detector=detector,
                          event=i, k=k)
        hit_modules = int(np.unique(module[np.bincount(ev.sensor_id, minlength=geometry.n_pmt) > 0]).size)
        r = {"event": i, "lambda_photons": lam, "w_bar": w_bar, "mu_rel_diff": rel,
             "photons_per_hit_pmt": lam / max(1, int(row["n_hit_pmts"])),
             "photons_per_hit_module": lam / max(1, hit_modules), "metrics": {}}
        for m in METRICS:
            obs = float(row[m])
            r["metrics"][m] = {**c[m], "observed": obs, "normalised": obs / c[m]["c"]}
        if with_split_half:
            r["split_half"] = split_half(ev, geometry, cfg, mu, detector=detector, event_index=i)
        rows.append(r)
    out = {"detector": detector, "skipped": False, "n_pmt": int(geometry.n_pmt),
           "n_modules": int(geometry.n_modules), "events": rows, "C": {}, "normalised_median": {},
           "observed_median": {}}
    for m in METRICS:
        out["C"][m] = float(np.median([r["metrics"][m]["c"] for r in rows]))
        out["normalised_median"][m] = float(np.median([r["metrics"][m]["normalised"] for r in rows]))
        out["observed_median"][m] = float(np.median([r["metrics"][m]["observed"] for r in rows]))
    return out


def main(argv=None) -> int:
    from . import ext_r2 as x
    from . import r2_generator as gen
    p = argparse.ArgumentParser()
    p.add_argument("--detector", required=True)
    p.add_argument("--rows", required=True, help="committed Gate V JSON (rows / gate_v.rows)")
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=K_DRAWS)
    a = p.parse_args(argv)
    doc = json.loads(Path(a.rows).read_text())
    rows = (doc.get("gate_v") or doc)["rows"]
    cfg = gen.load_r2_config()
    t0 = time.time()
    events = transported_events(a.detector, cfg, Path(a.cache))
    res = detector_ceiling(a.detector, rows, cfg, events=events, k=a.k)
    res.update({"addendum_sha256": ADDENDUM_SHA256, "seed": CEILING_SEED, "k": a.k,
                "rows_source": a.rows, "wall_s": time.time() - t0,
                "code_sha": lb.current_code_sha()})
    Path(x.check_write_path(a.out)).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: res.get(k) for k in ("detector", "skipped", "C", "normalised_median",
                                              "wall_s", "reason")}, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
