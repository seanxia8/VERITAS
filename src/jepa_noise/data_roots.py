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
#: r2-sparse W2: WCTE re-packed from the cached lucid_v1 LUCiD digits with the W1 window.
LUCID_V1_W2 = "lucid_v1_w2"
DATA_KINDS = (LUCID_V1, ANALYTIC_TOY, LUCID_V1_W2)
#: The manifest / pack ``dataset_kind`` each data kind must carry (T1.1: a mismatch raises).
PACK_KIND = {LUCID_V1: "lucid_mc_native", LUCID_V1_W2: "lucid_v1_w2"}
#: Readout window [ns] around the NHits trigger of each LUCiD data kind (WCTE).
WINDOW_NS = {LUCID_V1: (-20.0, 60.0), LUCID_V1_W2: (-110.0, 50.0)}
#: r2-sparse-sk J0: detectors with a lucid_v1_w2 re-pack and their windows (WCTE = WINDOW_NS, unchanged)
W2_WINDOW_NS = {"wcte": WINDOW_NS[LUCID_V1_W2], "juno": (-110.0, 350.0)}
DEFAULT_DATA_KIND = os.environ.get("JEPA_DATA_KIND", LUCID_V1)

LUCID_ROOT = Path("/ceph/dwong/jepa/data/lucid_v1")
W2_ROOT = Path("/ceph/dwong/jepa/data/lucid_v1_w2")
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
    if kind == LUCID_V1_W2 and detector not in W2_WINDOW_NS:
        raise ValueError("lucid_v1_w2 exists for WCTE and JUNO only")
    if kind == LUCID_V1_W2 and detector != "wcte":   # r2-sparse-sk J0: JUNO keeps the lucid_v1 layout
        return W2_ROOT / detector / f"{detector}_lucid_v1_w2.h5"
    if detector == "wcte":
        return wcte_probe_pack(kind)
    if kind == ANALYTIC_TOY:
        return TOY_R2_ROOT / detector / f"{detector}_r2.h5"
    return LUCID_ROOT / detector / f"{detector}_lucid_v1.h5"


def window_ns(detector: str, kind: str | None = None) -> tuple:
    """r2-sparse-sk: the readout window [ns] around the NHits trigger of a detector's LUCiD data.

    WCTE keeps ``WINDOW_NS`` of its data kind (unchanged); SK / HK / JUNO ``lucid_v1`` use their
    generator time rule's prompt window (SK [-20, +270], HK [-20, +460], JUNO [-20, +200] ns)."""
    kind = check(kind)
    if detector == "wcte":
        return WINDOW_NS[kind]
    if kind == LUCID_V1_W2 and detector in W2_WINDOW_NS:
        return W2_WINDOW_NS[detector]
    if kind != LUCID_V1:
        raise ValueError(f"no {kind} window for {detector}")
    from . import r2_generator as gen
    return tuple(float(v) for v in gen.time_rule_for(geometry(detector, kind)).prompt_window_ns)


def wcte_probe_pack(kind: str | None = None) -> Path:
    kind = check(kind)
    if kind == LUCID_V1_W2:
        return W2_ROOT / "wcte" / "wcte_lucid_v1_w2.h5"
    return TOY_WCTE_PROBE if kind == ANALYTIC_TOY else LUCID_ROOT / "wcte" / "wcte_lucid_v1.h5"


def sealed_pack(detector: str, kind: str | None = None) -> Path:
    kind = check(kind)
    if kind == LUCID_V1_W2:
        if detector not in W2_WINDOW_NS:
            raise ValueError("lucid_v1_w2 exists for WCTE and JUNO only")
        return W2_ROOT / detector / f"{detector}_lucid_v1_w2_sealed.h5"
    if kind != LUCID_V1:
        raise ValueError(f"no sealed pack for {kind}")
    return LUCID_ROOT / detector / f"{detector}_lucid_v1_sealed.h5"


def check_pack_kind(path, kind: str) -> str:
    """Raise unless the HDF5 pack at ``path`` holds data of ``kind`` (its ``dataset_kind`` attr)."""
    import h5py
    kind = check(kind)
    with h5py.File(path, "r") as f:
        found = f.attrs.get("dataset_kind")
    found = found.decode() if isinstance(found, bytes) else found
    if kind in PACK_KIND and found != PACK_KIND[kind]:
        raise ValueError(f"{path}: pack dataset_kind {found!r} is not {kind!r} ({PACK_KIND[kind]!r})")
    return found


def check_manifest_kind(manifest: dict, kind: str) -> None:
    """Raise unless a shard-directory manifest holds data of ``kind``."""
    kind = check(kind)
    if kind in PACK_KIND and manifest.get("dataset_kind") != PACK_KIND[kind]:
        raise ValueError(f"manifest dataset_kind {manifest.get('dataset_kind')!r} is not {kind!r}")


def wcte_train_shards(kind: str | None = None) -> Path:
    """Round-1 shard-directory layout (manifest.json + runs/<run_dir>/hits.npz)."""
    kind = check(kind)
    if kind == LUCID_V1_W2:
        return W2_ROOT / "wcte" / "wcte_train"
    return TOY_WCTE_TRAIN if kind == ANALYTIC_TOY else LUCID_ROOT / "wcte" / "wcte_train"


def geometry(detector: str, kind: str | None = None):
    """The geometry the data were generated with (LUCiD: frozen lucid_v1 geometry)."""
    from . import r2_geometry as rg
    if check(kind) == ANALYTIC_TOY:
        return rg.load_r2_geometry(detector)
    from . import lucid_mc_generator as mg
    return mg.lucid_geometry(detector, mg.load_config())[0]
