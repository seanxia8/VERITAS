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
        "release": {"research_use": "clarified_issue_6133100584", "redistribution": "not_approved",
                    "releasable": False, "evidence_stage": "development", "non_citable": True},
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


# --------------------------------------------------------------------------- response cells (12 Oct 2026)
#
# The claim-bearing LUCiD N families act on the readout chain (``readout.py``),
# not on the front-end electronics (frozen, R-1). A response cell moves one
# ``ReadoutParams`` field; every event is paired with the reference by photon
# list and readout seed. The consequence K is the frozen vertex fit of
# ``consequence.py``. Photons come from LUCiD after gate A0, or from ``toy``
# (development only, non-citable).


@dataclass(frozen=True)
class ResponseCellSpec:
    """One response cell: a family at a severity, as a change of the reference readout."""

    label: str
    family: str                         # "reference" or a key of response.RESPONSE_FAMILIES
    severity: float | None
    params: Any                         # readout.ReadoutParams
    origin: str = "N_struct"            # "clean" for the reference
    affected_groups: tuple[int, ...] = ()
    changed_fields: tuple[str, ...] = ()
    causal_control: str | None = None


def response_cells(n_channels: int, *, group_size: int = 64, group_method: str = "contiguous",
                   positions: np.ndarray | None = None, board_ids: np.ndarray | None = None,
                   reference: Any = None, affected: str = "alternate",
                   families: Sequence[str] | None = None) -> tuple[list[ResponseCellSpec], list[np.ndarray]]:
    """Reference + every response family at every severity of its development grid.

    Drifts (efficiency, gain, timing) act coherently per crate on the
    ``affected`` crates (``"alternate"``: every other crate; ``"all"``), so they
    are non-uniform and cannot be absorbed by a global energy or time scale.
    ``threshold_walk`` is detector-wide; ``module_loss`` kills ``k`` crates
    spread over the detector. Returns ``(cells, groups)``.
    """
    from .grouping import channel_groups
    from .readout import ReadoutParams, with_change
    from .response import RESPONSE_FAMILIES, dead_channels_from_groups, group_values

    ref = reference or ReadoutParams()
    groups = channel_groups(n_channels, group_size, positions=positions, method=group_method,
                            board_ids=board_ids)
    G = len(groups)
    if affected == "alternate":
        hit_groups = tuple(range(0, G, 2))
    elif affected == "all":
        hit_groups = tuple(range(G))
    else:
        raise ValueError("affected must be 'alternate' or 'all'.")
    cells = [ResponseCellSpec("reference", "reference", None, ref, origin="clean")]
    for fam in (families or RESPONSE_FAMILIES):
        grid = RESPONSE_FAMILIES[fam]["severity"]["grid"]
        for sev in grid:
            if fam == "efficiency_loss":
                prm, aff = with_change(ref, survival=group_values(groups, n_channels, sev, hit_groups, 1.0)), hit_groups
            elif fam == "gain_drift":
                prm, aff = with_change(ref, gain=group_values(groups, n_channels, sev, hit_groups, 1.0)), hit_groups
            elif fam == "timing_drift":
                prm, aff = with_change(ref, timing_offsets_ns=group_values(groups, n_channels, sev, hit_groups, 0.0)), hit_groups
            elif fam == "threshold_walk":
                prm, aff = with_change(ref, threshold_pe=float(sev)), tuple(range(G))
            elif fam == "module_loss":
                k = int(sev)
                if k > G:
                    continue
                dead_groups = tuple(int(round(i * G / k)) % G for i in range(k))
                prm = with_change(ref, dead_channels=tuple(int(c) for c in dead_channels_from_groups(groups, dead_groups)))
                aff = dead_groups
            else:
                raise KeyError(f"no cell builder for family {fam!r}")
            cells.append(ResponseCellSpec(f"N_struct:{fam}@{sev:g}", fam, float(sev), prm,
                                          affected_groups=tuple(int(a) for a in aff)))
    return cells, groups


def run_response_matrix(cells: Sequence[ResponseCellSpec], events: Sequence[Any], positions: np.ndarray,
                        out: Path | None = None, *, fit_config: Any = None, source: str = "toy", n_boot: int = 500,
                        lucid_commit_hash: str | None = None, calibration: Any = None) -> dict[str, Any]:
    """Run every response cell over the same events; return per-cell K and digit summaries.

    ``events`` need ``event_id``, ``vertex``, ``photon_sensor``, ``photon_time``
    and ``true_energy`` (``toy.ToyEvent`` or an adapter around LUCiD output).
    K has two endpoints (``consequence``): vertex error and relative energy
    error, with a frozen calibration from separate reference-fit events; the draft
    primary is ``K = max(K_vertex, K_energy)`` (D-K). The readout seed is
    the event id, so every cell is paired with the reference event by event.
    With ``out``, each cell is written as ``<cell>/digits.npz`` (ragged arrays
    with offsets), ``truth`` rows and ``provenance.json``, plus ``summary.json``.
    """
    from .consequence import (KAPPA_M_DEV, KAPPA_M_STATUS, cell_consequence, energy_calibration, energy_error,
                              fit_vertex, vertex_error)
    from .readout import readout_chain

    if not cells or cells[0].family != "reference":
        raise ValueError("the first cell must be the reference.")
    pos = np.asarray(positions, dtype=float)
    n_channels = pos.shape[0]
    errors: dict[str, list[float]] = {}
    e_errors: dict[str, list[float]] = {}
    e_errors_dc: dict[str, list[float]] = {}
    from .calibration import event_digits
    from .consequence import aggregate_endpoints
    from .group_response import validate_cell
    if calibration is None:
        raise ValueError("frozen calibration from separate reference-fit events required")
    calibration.validate_evaluation(events, pos, cells[0].params, fit_config)
    if any(getattr(ev, "endpoint_kind", None) != "point_source" for ev in events):
        raise ValueError("point-source endpoint cannot score tracks")
    for cell in cells:
        validate_cell(cell, cells[0].params)
    calib = calibration.energy_scale
    ref_label = cells[0].label
    summary: dict[str, Any] = {"source": source, "non_citable": True, "source_kind": source, "evidence_stage": "development", "physics_review_status": "pending", "protocol_freeze_id": None, "calibration_sha256": calibration.sha256, "n_events": len(events),
                               "kappa_m": {"value": KAPPA_M_DEV, "status": KAPPA_M_STATUS,
                                           "units": "ratio of mean error to the reference on the same events",
                                           "primary": "K = max(K_vertex, K_energy) (draft, D-K)"},
                               "cells": []}
    for cell in cells:
        errs, rows, sens, chg, tim, offs, qtot = [], [], [], [], [], [0], []
        digits = []
        for ev in events:
            d = event_digits(ev, n_channels, cell.params)
            digits.append(d)
            qtot.append(float(d.charge.sum()))
        eerr = [energy_error(q, ev.true_energy, calib) for q, ev in zip(qtot, events)]
        e_errors[cell.label] = eerr
        # known-dead-map correction (D-K item 17(c)): scale Q by the reference charge share of the
        # alive channels, measured on the reference cell — a correction the experiment would apply.
        dead = np.asarray(cell.params.dead_channels, dtype=int)
        alive_share = 1.0
        if dead.size:
            alive_share = 1.0 - float(np.asarray(calibration.channel_share)[dead].sum())
        eerr_dc = [energy_error(q / alive_share, ev.true_energy, calib) if alive_share > 0 else float("nan") for q, ev in zip(qtot, events)]
        e_errors_dc[cell.label] = eerr_dc
        for ev, d, ee, eedc in zip(events, digits, eerr, eerr_dc):
            f = fit_vertex(pos, d.sensor, d.time, d.charge, fit_config)
            e = vertex_error(f, ev.vertex)
            errs.append(e)
            rows.append({"event_id": int(ev.event_id), "cell": cell.label, "family": cell.family,
                         "origin": cell.origin, "severity": cell.severity, "vertex_error_m": e,
                         "energy_rel_error": ee, "energy_rel_error_deadcorr": eedc,
                         "q_total_pe": float(d.charge.sum()),
                         "fit_ok": f.ok, **{f"n_{k}": v for k, v in d.counts.items()},
                         "true_x": float(ev.vertex[0]), "true_y": float(ev.vertex[1]), "true_z": float(ev.vertex[2])})
            sens.append(d.sensor); chg.append(d.charge); tim.append(d.time); offs.append(offs[-1] + len(d))
        errors[cell.label] = errs
        kv = cell_consequence(errs, errors[ref_label], n_boot=n_boot)
        ke = cell_consequence(eerr, e_errors[ref_label], n_boot=n_boot)
        kedc = cell_consequence(eerr_dc, e_errors_dc[ref_label], n_boot=n_boot)
        K = aggregate_endpoints(kv["K"], ke["K"])
        entry = {"label": cell.label, "family": cell.family, "origin": cell.origin, "severity": cell.severity,
                 "moved": cell.params.moved_from(cells[0].params), "affected_groups": list(cell.affected_groups),
                 "causal_control": cell.causal_control, "changed_fields": list(cell.changed_fields),
                 "K": K, "K_vertex": kv, "K_energy": ke, "K_energy_deadcorr": kedc,
                 "K_deadcorr": aggregate_endpoints(kv["K"], kedc["K"]),
                 "harmful_dev": (bool(K >= KAPPA_M_DEV) if np.isfinite(K) else None),
                 "mean_digits": float(np.mean([r["n_digits"] for r in rows])),
                 "energy_calibration": calib}
        k = {"K": K, "vertex": kv, "energy": ke}
        summary["cells"].append(entry)
        if out is not None:
            d_ = Path(out) / cell.label
            d_.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d_ / "digits.npz", sensor=np.concatenate(sens), charge=np.concatenate(chg),
                                time=np.concatenate(tim), offsets=np.asarray(offs))
            prov = {"cell": cell.label, "family": cell.family, "origin": cell.origin, "severity": cell.severity,
                    "readout_params": cell.params.to_dict(), "moved": entry["moved"],
                    "causal_control": cell.causal_control, "changed_fields": list(cell.changed_fields),
                    "affected_groups": list(cell.affected_groups), "source": source,
                    "non_citable": True, "source_kind": source, "evidence_stage": "development", "physics_review_status": "pending", "protocol_freeze_id": None, "calibration_sha256": calibration.sha256, "lucid_commit": lucid_commit_hash,
                    "positions_hash": _sha1_array(pos), "consequence": k,
                    "release": {"research_use": "clarified_issue_6133100584", "redistribution": "not_approved"}}
            write_cell(Path(out), cell.label, rows, np.zeros((0,)), prov)
            (d_ / "traces.npy").unlink(missing_ok=True)          # hit-level cell: no waveform
    if out is not None:
        (Path(out) / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
    return summary


def toy_response_run(out: Path | None, *, n_events: int = 100, n_photons: int = 800,
                     families: Sequence[str] | None = None, group_size: int = 64) -> dict[str, Any]:
    """Legacy toy source with repaired independent calibration (non-citable).

    Historical saved results are preserved; this is not a byte replay of their
    former evaluation-fitted energy scale or old optimizer initialization.
    """
    from . import toy

    det = toy.cylinder_detector()
    events = [toy.point_source_event(det, e, n_photons=n_photons) for e in range(n_events)]
    cells, _ = response_cells(det.n_channels, group_size=group_size, families=families)
    from .calibration import fit_reference_calibration
    fit_events = [toy.point_source_event(det, 100000+e, n_photons=n_photons) for e in range(n_events)]
    calibration = fit_reference_calibration(fit_events, det.positions, cells[0].params)
    s = run_response_matrix(cells, events, det.positions, out, source="toy", calibration=calibration)
    s["toy"] = {"n_channels": det.n_channels, "radius_m": det.radius, "half_height_m": det.half_height,
                "n_photons": n_photons}
    if out is not None:
        (Path(out) / "summary.json").write_text(json.dumps(s, indent=2, default=_jsonable))
    return s


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
    ap.add_argument("--response-toy", action="store_true",
                    help="run the detector-response cells on the toy detector (development, non-citable; no LUCiD)")
    ap.add_argument("--n-events", type=int, default=100)
    ap.add_argument("--toy-photons", type=int, default=800)
    a = ap.parse_args(argv)
    if a.response_toy:
        s = toy_response_run(a.out, n_events=a.n_events, n_photons=a.toy_photons)
        for c in s["cells"]:
            kv, ke = c["K_vertex"], c["K_energy"]
            print(f"  {c['label']:34s} K={c['K']:.3f}  vtx {kv['K']:.3f} [{kv.get('K_lo', float('nan')):.2f},"
                  f"{kv.get('K_hi', float('nan')):.2f}]  E {ke['K']:.3f} [{ke.get('K_lo', float('nan')):.2f},"
                  f"{ke.get('K_hi', float('nan')):.2f}]  digits={c['mean_digits']:.0f}  harmful={c['harmful_dev']}")
        return 0
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
