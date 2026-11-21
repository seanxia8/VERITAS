# SPDX-License-Identifier: MIT
"""Default data roots (r2-lucid §L5.4): LUCiD data (``lucid_v1``) by default on branch lucid-mc.

The analytic toy stays selectable with ``--data-kind analytic_toy`` (or ``JEPA_DATA_KIND``). The
toy data are archived read-only (``/ceph/dwong/jepa/data/ARCHIVE_ANALYTIC_TOY_2026-11-21.md``).
"""
from __future__ import annotations

import os
from pathlib import Path

LUCID_V1 = "lucid_v1"
ANALYTIC_TOY = "analytic_toy"
DATA_KINDS = (LUCID_V1, ANALYTIC_TOY)
DEFAULT_DATA_KIND = os.environ.get("JEPA_DATA_KIND", LUCID_V1)

LUCID_ROOT = Path("/ceph/dwong/jepa/data/lucid_v1")
TOY_R2_ROOT = Path("/ceph/dwong/jepa/data/round2")
TOY_WCTE_TRAIN = Path("/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/lucid/"
                      "dev_train__F1__0.05__s20261017__n100000x1000")
TOY_WCTE_PROBE = Path("/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/wcte_regression/"
                      "wcte_probe_r2.h5")


def check(kind: str | None) -> str:
    kind = kind or DEFAULT_DATA_KIND
    if kind not in DATA_KINDS:
        raise ValueError(f"unknown data kind {kind!r}; choose from {DATA_KINDS}")
    return kind


def pack(detector: str, kind: str | None = None) -> Path:
    """The detector pack holding dev_train (SK/JUNO/HK) and the probe cohorts."""
    kind = check(kind)
    if detector == "wcte":
        return wcte_probe_pack(kind)
    if kind == ANALYTIC_TOY:
        return TOY_R2_ROOT / detector / f"{detector}_r2.h5"
    return LUCID_ROOT / detector / f"{detector}_lucid_v1.h5"


def wcte_probe_pack(kind: str | None = None) -> Path:
    return TOY_WCTE_PROBE if check(kind) == ANALYTIC_TOY else LUCID_ROOT / "wcte" / "wcte_lucid_v1.h5"


def wcte_train_shards(kind: str | None = None) -> Path:
    """Round-1 shard-directory layout (manifest.json + runs/<run_dir>/hits.npz)."""
    return TOY_WCTE_TRAIN if check(kind) == ANALYTIC_TOY else LUCID_ROOT / "wcte" / "wcte_train"


def geometry(detector: str, kind: str | None = None):
    """The geometry the data were generated with (LUCiD: frozen lucid_v1 geometry)."""
    from . import r2_geometry as rg
    if check(kind) == ANALYTIC_TOY:
        return rg.load_r2_geometry(detector)
    from . import lucid_mc_generator as mg
    return mg.lucid_geometry(detector, mg.load_config())[0]
