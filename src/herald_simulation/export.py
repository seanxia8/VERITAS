# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Parquet export with ``event_id`` as the pairing key and a full provenance record."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from ._hest import hest_commit


def _jsonable(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def write_cell(out_dir: Path, cell_label: str, rows: list[dict[str, Any]], traces: np.ndarray,
               provenance: dict[str, Any]) -> dict[str, Path]:
    """One cell → ``<cell>/truth.parquet`` (one row per event), ``traces.npy`` (n, C, N), ``provenance.json``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = Path(out_dir) / cell_label
    d.mkdir(parents=True, exist_ok=True)
    flat = [{k: (json.dumps(_jsonable(v)) if isinstance(v, (dict, list, tuple, np.ndarray)) else _jsonable(v))
             for k, v in r.items()} for r in rows]
    table = pa.Table.from_pylist(flat)
    pq.write_table(table, d / "truth.parquet")
    np.save(d / "traces.npy", np.asarray(traces, dtype=np.float32))
    prov = {"herald_simulation_version": __version__, "hest_commit": hest_commit(), "cell": cell_label, **_jsonable(provenance)}
    (d / "provenance.json").write_text(json.dumps(prov, indent=2))
    return {"truth": d / "truth.parquet", "traces": d / "traces.npy", "provenance": d / "provenance.json"}
