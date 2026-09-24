# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Build a paired event set: one reference cell, then one factor moved per cell.

    PYTHONPATH=src python -m herald_simulation.simulate --out runs/herald_pilot --n-events 8 --qp-fraction 0.02

Cells: reference (HeRALD_v1, ER at 1 keV, TES_HERALD_V1); G — the 24→1
monolithic contrast and a 2-sensor split; Σ — bath correlation, pickup modes,
SQUID knee, mains; N-structural — sensor loss, gain drift, timing jitter;
E — NR at the same energy, half and double energy; U — a WIMP spectrum. Every
cell reuses the same ``event_id`` list, so every row pairs with the reference.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .events import evaporate
from .export import write_cell
from .geometry import HeraldGeometry, shipped
from .interventions import Structural, structural_cells
from .noise import NoiseSpec, add_noise, sigma_cells
from .strata import EventFamily, event_cells, reference_family, undeclared_cells
from .traces import TraceConfig, clean_traces


@dataclass(frozen=True)
class Cell:
    label: str
    moved: str
    geometry: HeraldGeometry
    noise: NoiseSpec
    family: EventFamily
    structural: Structural = Structural("none")
    trace: TraceConfig = TraceConfig()


def reference_cell(qp_fraction: float = 1.0) -> Cell:
    trace = replace(TraceConfig(), amplitude_scale=1.0 / qp_fraction)
    return Cell("reference", "none", shipped("HeRALD_v1"), NoiseSpec(), reference_family(), trace=trace)


def all_cells(ref: Cell) -> list[Cell]:
    cells: list[Cell] = []
    for g in (shipped("HeRALD_v1_monolithic"), shipped("HeRALD_UMass_splitCPD")):
        cells.append(replace(ref, geometry=g, label=f"geometry:{g.name}", moved="geometry"))
    for nm in sigma_cells(ref.noise):
        cells.append(replace(ref, noise=nm, label=f"sigma_cov:{nm.label}", moved="sigma_cov"))
    for st in structural_cells():
        cells.append(replace(ref, structural=st, label=f"sigma_struct:{st.kind}", moved="sigma_struct"))
    for fam in event_cells(ref.family):
        cells.append(replace(ref, family=fam, label=f"event:{fam.label}", moved="event"))
    for fam in undeclared_cells():
        cells.append(replace(ref, family=fam, label=f"undeclared:{fam.label}", moved="undeclared"))
    return cells


def run_cell(cell: Cell, event_ids: np.ndarray, qp_fraction: float, out: Path | None = None) -> dict[str, Any]:
    t0 = time.time()
    det = cell.geometry.detector()
    rows, traces = [], []
    sim = cell.trace.simulator()
    for eid in event_ids:
        ev = evaporate(cell.geometry, int(eid), cell.family.energy(int(eid)), cell.family.interaction,
                       vertex_cm=cell.family.vertex_cm, qp_fraction=qp_fraction, detector=det)
        X, tmeta = clean_traces(ev, cell.trace, sim)
        X = cell.structural.apply(X, ev.seed)
        X, nmeta = add_noise(X, cell.noise, cell.trace.sampling_frequency, ev.seed)
        rows.append({"event_id": ev.event_id, "geometry": ev.geometry_name, "geometry_hash": ev.geometry_hash,
                     "cell": cell.label, "moved": cell.moved, "seed": ev.seed, **ev.truth(),
                     "n_qp_simulated": ev.meta["n_qp_simulated"], "qp_fraction": ev.meta["qp_fraction"],
                     "amplitude_adc": tmeta["amplitude_adc"], "kappa_floor": nmeta["kappa"],
                     "mean_offdiag_corr": nmeta["mean_offdiag_corr"], "structural": cell.structural.to_dict()})
        traces.append(X)
    traces_arr = np.stack(traces)
    prov = {"geometry": {"name": cell.geometry.name, "hash": cell.geometry.geometry_hash, "n_sensors": cell.geometry.n_sensors,
                         "positions_cm": cell.geometry.positions_cm, "params": cell.geometry.params},
            "noise": cell.noise.provenance(), "family": cell.family.to_dict(), "structural": cell.structural.to_dict(),
            "trace": cell.trace.to_dict(), "event_ids": event_ids.tolist(), "qp_fraction": qp_fraction,
            "seconds": time.time() - t0, "implied_covariance": nmeta["implied_covariance"]}
    paths = write_cell(out, cell.label, rows, traces_arr, prov) if out is not None else {}
    return {"label": cell.label, "n_events": len(rows), "shape": traces_arr.shape, "seconds": prov["seconds"],
            "mean_detected": float(np.mean([r["n_detected"] for r in rows])), "kappa_floor": float(np.mean([r["kappa_floor"] for r in rows])),
            "paths": {k: str(v) for k, v in paths.items()}}


def build_event_set(out: Path, n_events: int = 8, qp_fraction: float = 0.02, first_id: int = 1, only: list[str] | None = None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    ref = reference_cell(qp_fraction)
    event_ids = np.arange(first_id, first_id + n_events)
    cells = [ref] + all_cells(ref)
    if only:
        cells = [c for c in cells if c.label == "reference" or any(c.label.startswith(o) for o in only)]
    summary = {"n_events": n_events, "qp_fraction": qp_fraction, "cells": []}
    for cell in cells:
        r = run_cell(cell, event_ids, qp_fraction, out)
        summary["cells"].append(r)
        print(f"  {cell.label:34s} {str(r['shape']):16s} detected/event {r['mean_detected']:8.1f}  kappa_floor {r["kappa_floor"]:6.2f}  {r['seconds']:6.1f}s")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("runs/herald_pilot"))
    ap.add_argument("--n-events", type=int, default=8)
    ap.add_argument("--qp-fraction", type=float, default=0.02, help="thin the QP population for cost; amplitude rescaled")
    ap.add_argument("--only", nargs="*", default=None, help="cell-label prefixes to run (reference always runs)")
    a = ap.parse_args(argv)
    build_event_set(a.out, a.n_events, a.qp_fraction, only=a.only)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
