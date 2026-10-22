# SPDX-License-Identifier: MIT
"""A10 PhotonSim vs analytic-Cherenkov validation harness.

The analytic particle-bomb model in :mod:`jepa_noise.lucid_bombs` is a simplified
track/shower Cherenkov source.  A10 requires it to reproduce real PhotonSim μ⁻
events (hit pattern, light yield and first-arrival time) within pre-declared
tolerances before the mixed grid.  This module contains only the comparison and
reporting logic; the host adapter supplies events with an explicit schema:

    event_id, domain, vertex_m, direction_m, energy_MeV, deposited_MeV,
    contained, track_length_m, sensor_id, time_ns, wavelength_nm

``sensor_id/time_ns/wavelength_nm`` are **after** transport (incident photons at
the PMTs), not raw emission points.  The host adapter is responsible for running
the pinned LUCiD ``PromptTransport`` (or the PMT PhotonSim reader) and for
recording its provenance.  JSON only; no data is written outside ``--out``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import lucid_bombs as lb


@dataclass(frozen=True)
class ValidationEvent:
    event_id: int
    domain: str
    vertex_m: np.ndarray
    sensor_id: np.ndarray
    time_ns: np.ndarray
    wavelength_nm: np.ndarray
    direction_m: np.ndarray | None = None
    energy_MeV: float = float("nan")
    deposited_MeV: float = float("nan")
    contained: bool = False
    track_length_m: float = float("nan")
    yield_photons: float = float("nan")

    def __post_init__(self):
        if self.domain not in lb.DOMAINS:
            raise ValueError(f"unknown validation domain {self.domain!r}")
        for name, value in (("vertex_m", self.vertex_m), ("sensor_id", self.sensor_id),
                            ("time_ns", self.time_ns), ("wavelength_nm", self.wavelength_nm)):
            array = np.asarray(value)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"validation event {name} must be finite")
        if self.time_ns.shape != self.sensor_id.shape or self.wavelength_nm.shape != self.sensor_id.shape:
            raise ValueError("sensor_id, time_ns and wavelength_nm must share a shape")


def load_event_npz(path: Path | str) -> ValidationEvent:
    with np.load(Path(path), allow_pickle=False) as data:
        values = {name: data[name] for name in data.files}
    return ValidationEvent(
        event_id=int(np.asarray(values["event_id"])),
        domain=str(np.asarray(values["domain"])),
        vertex_m=np.asarray(values["vertex_m"], float).reshape(3),
        sensor_id=np.asarray(values["sensor_id"], np.int64),
        time_ns=np.asarray(values["time_ns"], float),
        wavelength_nm=np.asarray(values["wavelength_nm"], float),
        direction_m=(None if "direction_m" not in values else np.asarray(values["direction_m"], float).reshape(3)),
        energy_MeV=float(np.asarray(values.get("energy_MeV", np.nan))),
        deposited_MeV=float(np.asarray(values.get("deposited_MeV", np.nan))),
        contained=bool(np.asarray(values.get("contained", False))),
        track_length_m=float(np.asarray(values.get("track_length_m", np.nan))),
        yield_photons=float(np.asarray(values.get("yield_photons", np.nan))),
    )


def _observed(event: ValidationEvent, geometry, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = geometry.n_pmt
    counts = np.bincount(event.sensor_id, minlength=n).astype(float)
    qe = lb.qe_at(event.wavelength_nm, cfg)
    detected = np.bincount(event.sensor_id, weights=qe * float(cfg.ce), minlength=n)
    first = np.full(n, np.inf)
    np.minimum.at(first, event.sensor_id, event.time_ns)
    return counts, detected, first


def compare_event(geometry, cfg: lb.GeneratorConfig, event: ValidationEvent) -> dict:
    """Compare one transported PhotonSim event with the analytic particle model."""
    counts, detected, observed_first = _observed(event, geometry, cfg)
    if event.domain == lb.DOMAIN_PHOTON:
        if not np.isfinite(event.yield_photons):
            raise ValueError("photon validation event needs yield_photons")
        qe = np.full(geometry.n_pmt, float(lb.qe_at(np.array([450.0]), cfg)[0]))
        mu = lb.expected_charge(geometry, event.vertex_m, event.yield_photons, qe,
                                lb.LAMBDA0_M, cfg)
        analytic_first = lb.propagation_time_ns(geometry, event.vertex_m, cfg)
        truth = None
    else:
        if event.direction_m is None or not (np.isfinite(event.energy_MeV)
                                            and np.isfinite(event.deposited_MeV)):
            raise ValueError("particle validation event needs direction, energy and deposited energy")
        truth = lb.ParticleTruth(domain=event.domain, vertex_m=event.vertex_m,
                                 direction=event.direction_m, energy_MeV=event.energy_MeV,
                                 deposited_MeV=event.deposited_MeV, contained=event.contained,
                                 track_length_m=event.track_length_m)
        qe_mean = float(np.mean(lb.qe_at(event.wavelength_nm, cfg))) if event.wavelength_nm.size else 0.2
        mu, analytic_first = lb.expected_charge_particle(
            geometry, truth, np.full(geometry.n_pmt, qe_mean), lb.LAMBDA0_M, cfg)
    total_expected, total_observed = float(mu.sum()), float(detected.sum())
    ratio = total_observed / total_expected if total_expected > 0 else float("nan")
    keep = (counts > 0) & (mu > 0)
    pattern = float(np.corrcoef(mu[keep], detected[keep])[0, 1]) if keep.sum() >= 3 else float("nan")
    hit_mask = (counts > 0) & np.isfinite(analytic_first) & np.isfinite(observed_first)
    time_residuals = observed_first[hit_mask] - analytic_first[hit_mask]
    return {
        "event_id": int(event.event_id), "domain": event.domain,
        "n_photons": int(event.sensor_id.size), "n_hit_pmts": int((counts > 0).sum()),
        "expected_detected": total_expected, "observed_detected": total_observed,
        "yield_ratio": ratio, "pattern_pearson": pattern,
        "time_median_abs_ns": float(np.median(np.abs(time_residuals))) if time_residuals.size else None,
        "time_p95_abs_ns": float(np.percentile(np.abs(time_residuals), 95)) if time_residuals.size else None,
        "truth": None if truth is None else {"energy_MeV": truth.energy_MeV,
                                             "deposited_MeV": truth.deposited_MeV,
                                             "contained": truth.contained},
    }


def validate(geometry, cfg: lb.GeneratorConfig, events, *, yield_tolerance=(0.5, 2.0),
             min_pattern=0.8, max_time_median_ns=5.0) -> dict:
    rows = [compare_event(geometry, cfg, event) for event in events]
    if not rows:
        raise ValueError("no validation events")
    ratios = np.array([r["yield_ratio"] for r in rows], dtype=float)
    patterns = np.array([r["pattern_pearson"] for r in rows], dtype=float)
    times = np.array([r["time_median_abs_ns"] for r in rows if r["time_median_abs_ns"] is not None],
                     dtype=float)
    passed = bool(np.all(np.isfinite(ratios)) and np.all((ratios >= yield_tolerance[0])
                                                         & (ratios <= yield_tolerance[1]))
                  and np.all(np.isfinite(patterns)) and np.all(patterns >= min_pattern)
                  and (times.size == 0 or np.all(times <= max_time_median_ns)))
    return {"schema": "jepa-cherenkov-validation-v0", "events": len(rows),
            "thresholds": {"yield_tolerance": list(yield_tolerance), "min_pattern": min_pattern,
                           "max_time_median_ns": max_time_median_ns},
            "summary": {"yield_ratio_median": float(np.median(ratios)),
                        "pattern_pearson_median": float(np.median(patterns)),
                        "time_median_abs_ns_median": (float(np.median(times)) if times.size else None)},
            "passed": passed, "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--geometry", required=True, help="WCTE_geom_config.json or .npz")
    parser.add_argument("--config", required=True, help="lucid_bombs_v1_mixed.json")
    parser.add_argument("--events", required=True, help="directory of per-event .npz files")
    parser.add_argument("--out", required=True, help="output JSON")
    args = parser.parse_args(argv)
    geometry = lb.load_wcte_geometry(args.geometry)
    cfg = lb.load_generator_config(args.config)
    paths = sorted(Path(args.events).glob("*.npz"))
    result = validate(geometry, cfg, (load_event_npz(path) for path in paths))
    Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result["summary"], indent=1))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
