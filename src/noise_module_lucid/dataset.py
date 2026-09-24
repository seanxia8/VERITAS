# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""LUCiD-arm dataset driver: cells, intervention matrix, writer and provenance.

Mirrors ``herald_simulation.export``: one cell → ``truth.parquet`` (one row per
event), ``traces.npy`` ``(n, C, N)`` and ``provenance.json``. The difference is
the substrate: here the "event" is a LUCiD photoelectron waveform (converted to
mV by ``units.to_mv``) and the noise is the front-end preset.

The driver is split so the pipeline can be exercised **without LUCiD**:

* :func:`run_cell` takes precomputed photoelectron waveforms (pure numpy);
* :func:`run_lucid_cell` builds the LUCiD simulator lazily and calls
  :func:`run_cell`. Nothing in this module imports ``lucid`` at import time, so
  the licence gate (A0) does not block the package.

Provenance records LUCiD commit, geometry hash, placed-sensor count, window,
readout (with its calibration flag), preset version, implied/realized
covariance, kappa, grouping and the release status.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from noise_module import to_jsonable

from . import __version__, interventions
from .adapter import add_pmt_noise, kappa
from .presets import PROVENANCE
from .readouts import PMT_1GHZ, Readout
from .units import to_mv

# --------------------------------------------------------------------------- specs


@dataclass(frozen=True)
class EventSpec:
    """One physical event: a source type and its parameters (position, energy, ...)."""

    event_id: int
    source: str = "isotropic"          # "isotropic" | "track" | "cascade" | "external"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CellSpec:
    """One experiment cell: geometry × event × intervention, on one readout."""

    label: str
    moved: str                          # "reference" | "sigma_cov" | "sigma_struct" | "event" | "geometry" | "undeclared"
    geometry: str
    event: EventSpec
    intervention: str | None = None
    window_ns: float = 512.0
    bin_width_ns: float = 1.0
    group_size: int = 64
    group_method: str = "contiguous"
    board_ids: np.ndarray | None = None
    readout: Readout = PMT_1GHZ
    geometry_hash: str | None = None
    preset_override: tuple[dict, dict] | None = None


# --------------------------------------------------------------------------- helpers


def _sha1_array(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


def _jsonable(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    return to_jsonable(o)


def lucid_commit(lucid_path: str | Path | None = None) -> str | None:
    """The LUCiD clone's HEAD commit, if a clone is present."""
    import os

    path = Path(lucid_path or os.environ.get("LUCID_PATH", "external/LUCiD"))
    if not (path / ".git").exists():
        return None
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def write_cell(out_dir: Path, cell_label: str, rows: list[dict[str, Any]], traces: np.ndarray,
               provenance: dict[str, Any]) -> dict[str, Path]:
    """One cell → ``<cell>/truth.parquet`` (or ``.csv``), ``traces.npy``, ``provenance.json``."""
    d = Path(out_dir) / cell_label
    d.mkdir(parents=True, exist_ok=True)
    flat = [{k: (json.dumps(_jsonable(v)) if isinstance(v, (dict, list, tuple, np.ndarray)) else _jsonable(v))
             for k, v in r.items()} for r in rows]
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(flat), d / "truth.parquet")
        truth_path = d / "truth.parquet"
    except Exception:                                   # pyarrow optional
        import csv

        truth_path = d / "truth.csv"
        with truth_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            writer.writeheader()
            writer.writerows(flat)
    np.save(d / "traces.npy", np.asarray(traces, dtype=np.float32))
    prov = {"noise_module_lucid_version": __version__, "cell": cell_label, **_jsonable(provenance)}
    (d / "provenance.json").write_text(json.dumps(prov, indent=2))
    return {"truth": truth_path, "traces": d / "traces.npy", "provenance": d / "provenance.json"}


# --------------------------------------------------------------------------- core


def _preset_for(cell: CellSpec):
    """Return ``(kind, preset_or_fn, base, crate)`` for a cell's intervention."""
    if cell.intervention is None:
        base, crate = cell.preset_override or cell.readout.base_crate()
        return "reference", None, base, crate
    kind, fn = interventions.INTERVENTIONS[cell.intervention]
    if kind == "preset":
        base, crate = fn()
    else:
        base, crate = cell.preset_override or cell.readout.base_crate()
    return kind, fn, base, crate


def run_cell(cell: CellSpec, events_pe: Sequence[tuple[EventSpec, np.ndarray]],
             out: Path | None = None, *, positions: np.ndarray | None = None,
             channel_gains: np.ndarray | None = None, lucid_commit_hash: str | None = None,
             extra_provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one cell over ``(EventSpec, pe_waveform)`` pairs and (optionally) write it.

    ``events_pe`` is a sequence of ``(EventSpec, (C, N) photoelectron array)``.
    Pure numpy — no LUCiD — so the pipeline is testable and the substrate can be
    any source.
    """
    if not events_pe:
        raise ValueError("run_cell needs at least one (EventSpec, waveform) pair.")
    ro = cell.readout
    kind, fn, base, crate = _preset_for(cell)
    fs = ro.sampling_frequency_hz
    rows, traces, implied, floors = [], [], None, []
    groups = None
    for ev, pe in events_pe:
        sig_mv = to_mv(pe, ro.spe())
        trace, groups = add_pmt_noise(sig_mv, (base, crate), seed=int(ev.event_id),
                                      group=cell.group_size, positions=positions,
                                      group_method=cell.group_method, board_ids=cell.board_ids,
                                      channel_gains=channel_gains)
        if kind == "trace":
            trace = fn(trace, fs)
        kappas = [kappa(m) for m in groups]
        implied = groups[0]["implied_covariance"]
        floors.append(float(np.mean(kappas)))
        rows.append({
            "event_id": int(ev.event_id), "geometry": cell.geometry,
            "geometry_hash": cell.geometry_hash, "cell": cell.label, "moved": cell.moved,
            "source": ev.source, "intervention": cell.intervention,
            "n_detected": int(np.count_nonzero(pe.sum(axis=1))), "total_pe": float(pe.sum()),
            "kappa_floor": float(np.mean(kappas)),
            "mean_offdiag_corr": float(np.mean([m["mean_offdiag_corr"] for m in groups])),
            **{f"p_{k}": v for k, v in ev.params.items() if np.isscalar(v)},
        })
        traces.append(trace)
    traces_arr = np.stack(traces)
    grouping = {"method": cell.group_method, "group_size": cell.group_size,
                "n_groups": len(groups) if groups is not None else None}
    if positions is not None and groups is not None:
        from .validation import grouping_report

        grouping.update(grouping_report(np.asarray(positions),
                                        [np.asarray(m["channel_indices"]) for m in groups]))
    prov = {
        "geometry": {"name": cell.geometry, "hash": cell.geometry_hash,
                     "n_sensors": int(events_pe[0][1].shape[0]),
                     "positions_hash": _sha1_array(positions) if positions is not None else None},
        "window_ns": cell.window_ns, "bin_width_ns": cell.bin_width_ns,
        "readout": ro.provenance(), "preset": PROVENANCE,
        "intervention": cell.intervention, "moved": cell.moved,
        "grouping": grouping,
        "implied_covariance": implied,
        "kappa_floor_mean": float(np.mean(floors)) if floors else None,
        "lucid_commit": lucid_commit_hash,
        "release": {"gate_A0_lucid_licence": "open",
                    "releasable": bool(ro.calibrated) and lucid_commit_hash is not None},
        "extra": _jsonable(extra_provenance) if extra_provenance else {},
    }
    paths = write_cell(out, cell.label, rows, traces_arr, prov) if out is not None else {}
    return {"label": cell.label, "n_events": len(rows), "shape": traces_arr.shape,
            "kappa_floor": float(np.mean(floors)) if floors else None,
            "paths": {k: str(v) for k, v in paths.items()}}


# --------------------------------------------------------------------------- LUCiD driver


def lucid_available(lucid_path: str | Path | None = None) -> bool:
    import os

    path = Path(lucid_path or os.environ.get("LUCID_PATH", "external/LUCiD"))
    return (path / "lucid").is_dir()


def build_lucid_simulator(geometry: str, *, window_ns: float = 512.0, bin_width_ns: float = 1.0,
                          n_photons: int = 50_000, lucid_path: str | Path | None = None,
                          particle: str = "muon", wavelength_mode: bool = False):
    """Build ``(detector, simulator)`` from the LUCiD clone (lazy import)."""
    import os
    import sys

    path = Path(lucid_path or os.environ.get("LUCID_PATH", "external/LUCiD"))
    if not (path / "lucid").is_dir():
        raise FileNotFoundError(f"LUCiD clone not found at {path}; set LUCID_PATH")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator

    geom_json = path / "config" / f"{geometry}_geom_config.json"
    phys_json = path / "config" / f"{geometry}_physics_config.json"
    det = generate_detector(str(geom_json))
    sim = setup_event_simulator(
        str(geom_json), n_photons, temperature=None, K=6, is_calibration=True,
        hit_mode="waveform", physics_config=str(phys_json) if phys_json.exists() else None,
        default_detector_params=phys_json.exists(), wavelength_mode=wavelength_mode,
        waveform_config=dict(window_ns=window_ns, bin_width_ns=bin_width_ns))
    return det, sim


def lucid_isotropic_source(position, intensity: int):
    from lucid.sources import isotropic_source

    return isotropic_source(position=list(position), intensity=int(intensity))


def run_lucid_cell(cell: CellSpec, out: Path | None = None, *, n_photons: int = 50_000,
                   lucid_path: str | Path | None = None, source_factory: Callable | None = None,
                   channel_gains: np.ndarray | None = None) -> dict[str, Any]:
    """Build the LUCiD event for a cell, add the front end, and write it.

    The default source is the isotropic flasher (self-contained); pass
    ``source_factory(sim, event)`` for track/cascade or a PhotonSim source.
    """
    import jax

    det, sim = build_lucid_simulator(cell.geometry, window_ns=cell.window_ns,
                                     bin_width_ns=cell.bin_width_ns, n_photons=n_photons,
                                     lucid_path=lucid_path)
    positions = np.asarray(det.all_points)
    ev = cell.event
    if source_factory is not None:
        source = source_factory(sim, ev)
    elif ev.source == "isotropic":
        source = lucid_isotropic_source(ev.params.get("position", [0.0, 0.0, 0.0]),
                                        ev.params.get("intensity", n_photons))
    else:
        raise ValueError(f"source {ev.source!r} needs an explicit source_factory")
    pe = np.asarray(sim(source, jax.random.PRNGKey(int(ev.event_id)))[0])
    return run_cell(cell, [(ev, pe)], out, positions=positions, channel_gains=channel_gains,
                    lucid_commit_hash=lucid_commit(lucid_path))


# --------------------------------------------------------------------------- matrix


def reference_cell(geometry: str = "WCTE_like", event: EventSpec | None = None,
                   readout: Readout = PMT_1GHZ, **kwargs) -> CellSpec:
    event = event or EventSpec(1, "isotropic", {"position": [0.0, 0.0, 0.0], "intensity": 50_000})
    return CellSpec("reference", "reference", geometry, event, readout=readout, **kwargs)


def intervention_matrix(geometry: str = "WCTE_like", event: EventSpec | None = None,
                        readout: Readout = PMT_1GHZ, window_ns: float = 512.0,
                        **kwargs) -> list[CellSpec]:
    """Reference + declared N families (covariance-type and structural)."""
    ref = reference_cell(geometry, event, readout, window_ns=window_ns, **kwargs)
    cells = [ref]
    for name in ("clock_x5", "clock_deterministic", "broadband_common_mode_30"):
        cells.append(replace(ref, label=f"sigma_cov:{name}", moved="sigma_cov", intervention=name))
    for name in ("quantise_1mV", "aperture_jitter_50ps", "cable_delay_10ns", "alias_fold_4"):
        cells.append(replace(ref, label=f"sigma_struct:{name}", moved="sigma_struct", intervention=name))
    return cells


def covariance_cells(geometry: str = "WCTE_like", event: EventSpec | None = None,
                     readout: Readout = PMT_1GHZ, window_ns: float = 32_000.0,
                     **kwargs) -> list[CellSpec]:
    """Covariance cells on a long window where the kappa floor is usable.

    Uses the long-window preset (DC-DC lines, extended 1/f); the window is
    asserted against the switching-line resolution.
    """
    from copy import deepcopy

    from .adapter import crate_preset
    from .presets import long_window_components

    if window_ns < 16_000.0:
        raise ValueError("covariance cells need window_ns >= 16000 (16 us) to resolve the switching lines")
    private, shared = long_window_components(window_ns)
    common = deepcopy(shared) + [{"type": "white", "scale": 0.43, "name": "broadband_common_mode"}]
    make = lambda comps: crate_preset(  # noqa: E731
        shared=comps, private=private, sampling_frequency=readout.sampling_frequency_hz,
        noise_power=readout.rms_mv**2)
    ref = replace(reference_cell(geometry, event, readout, window_ns=window_ns, **kwargs),
                  preset_override=make(shared))
    cells = [replace(ref, label="cov_reference")]
    cells.append(replace(ref, label="cov_sigma:broadband_common_mode", moved="sigma_cov",
                         preset_override=make(common)))
    return cells


def build_dataset(out: Path, cells: Sequence[CellSpec], *, n_photons: int = 50_000,
                  lucid_path: str | Path | None = None) -> dict[str, Any]:
    """Run every cell with the LUCiD driver and write the dataset tree."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"n_cells": len(cells), "cells": []}
    for cell in cells:
        r = run_lucid_cell(cell, out, n_photons=n_photons, lucid_path=lucid_path)
        summary["cells"].append(r)
        print(f"  {cell.label:34s} {str(r['shape']):16s} kappa_floor {r['kappa_floor']:.2f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Produce the LUCiD-arm dataset (intervention matrix + covariance cells).")
    ap.add_argument("--out", type=Path, default=Path("runs/lucid_pilot"))
    ap.add_argument("--geometry", default="WCTE_like")
    ap.add_argument("--n-photons", type=int, default=50_000)
    ap.add_argument("--event-id", type=int, default=1)
    ap.add_argument("--covariance", action="store_true", help="also run the long-window covariance cells")
    ap.add_argument("--cov-window-ns", type=float, default=32_000.0)
    ap.add_argument("--lucid-path", default=None)
    a = ap.parse_args(argv)
    if not lucid_available(a.lucid_path):
        ap.error("LUCiD clone not found; clone it to external/LUCiD or set LUCID_PATH")
    ev = EventSpec(a.event_id, "isotropic", {"position": [0.0, 0.0, 0.0], "intensity": a.n_photons})
    cells = intervention_matrix(a.geometry, ev)
    if a.covariance:
        cells = cells + covariance_cells(a.geometry, ev, window_ns=a.cov_window_ns)
    summary = build_dataset(a.out, cells, n_photons=a.n_photons, lucid_path=a.lucid_path)
    print(json.dumps({"out": str(a.out), "n_cells": summary["n_cells"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
