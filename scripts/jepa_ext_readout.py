# SPDX-License-Identifier: MIT
"""Q1/Q2 readout-matrix extension CLI (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-01_READOUT.md`` (revision 2), §3.5-§3.7.

Modes
-----
dry-run   validate the checkpoint manifest (copies + both hashes), the probe-set
          preflight and the planned items x forms; write ``plan.json`` and
          ``items.json`` (and the verified checkpoint copies) under ``<ext>``.
matrix    one item group (≤ 4 items) on an L40S GPU: stream the probe sets once
          per representation, write every §2 metric for every form, cache the
          event/token rows for ``nonlinear``, and write ``reproduction_<group>.json``
          (exit 2 on a §4.2 mismatch).
nonlinear CPU: ridge + MLP probes on the identical cached rows (Q2-a/b/c) and the
          CKA rows (Q2-d).
report    merge the two matrix jobs and ``nonlinear`` into
          ``results/jepa/20261101_readout/`` with a ``registered: false`` README,
          the §6 readings and ``reproduction.json``.

Nothing outside ``<ext>`` (``/ceph/dwong/jepa/returns/ext_readout_20261101/``) is
written by dry-run/matrix/nonlinear; ``report`` writes only to the results
directory.  No registered file, config, criterion, prediction or verdict is read
for writing or changed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_noise import ext_readout as er  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402
from jepa_noise import point_encoder as pe  # noqa: E402

EXTENSION = er.EXTENSION
CONFIG_PATH = ROOT / "src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json"
GENERATOR_PATH = ROOT / "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json"
CONFIG_SHA256 = "e6730ac14ddf475b857a0f8b07407f8b54e487e2e7af827fd00bcecc5268e034"
GENERATOR_SHA256 = "90ac1601c40171eeb7c669a08df1fe6e6b5b332fee2a3cca689ae057a51dfe88"
GEOMETRY_DEFAULT = "/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json"
DATA_ROOT_DEFAULT = "/ceph/dwong/jepa/data/ext_pregrid_dev_20261025"
PROVENANCE = ROOT / "results/jepa/20261025_pregrid_checks/provenance.json"
D2_SHA256SUMS = ROOT / "results/jepa/20261029_d2_partial/SHA256SUMS"
#: §3.4 provenance manifest sha256 of the two probe sets (also in provenance.json).
PROBE_MANIFESTS = {"dev_probe_train": "093ea8949b6f079841f389775bfab771b0e2d9d9cc20f4dff707bea1532ecfd3",
                   "dev_probe_test": "c11edc711833dddc9520cb69435608ad4654f66ef80bcf49810d32fa667a8c5a"}
#: §3.4/§3.5 the probe sets are the only data used; dev_train is never built here.
DEV_TRAIN_MANIFEST = "b1b3f4f401bcee48c49c347b67d7f5cbd4410213f08ef8b9d2848edc9f2cd13a"

#: §3.3 checkpoint manifest (read-only).  The across items are verified through
#: their sibling ``ext_summary.json`` sha256 in the D2 ``SHA256SUMS`` and then hashed.
ITEMS = {
    "old9000": {"arm": "sigreg_jepa", "split": "within", "seed": 9000, "beta": None,
                "source": "/work/dwong/jepa_a10_work/dev_step2/runs/"
                          "dev_s005_within__sigreg_jepa__s9000/final.pt",
                "sha256": "b1632266f2f8ca9fd49f9a06383734fda77432f5247294e79ed65bc1936f7503"},
    "old9001": {"arm": "sigreg_jepa", "split": "within", "seed": 9001, "beta": None,
                "source": "/work/dwong/jepa_a10_work/old_s9001/runs/"
                          "dev_s005_within__sigreg_jepa__s9001/final.pt",
                "sha256": "14e3590f9ed6649d86f3de751a56c3590bb748f0b4a15984bb056373bff97ceb"},
    "fix9000": {"arm": "sigreg_jepa", "split": "within", "seed": 9000, "beta": None,
                "source": "/work/dwong/jepa_a10_work/fixed_s9000/runs/"
                          "dev_s005_within__sigreg_jepa__s9000/final.pt",
                "sha256": "2d912a2deef7eb74137405062c0bb55d82ad3d1cf97d4b37e02ba381088ef714"},
    "fix9001": {"arm": "sigreg_jepa", "split": "within", "seed": 9001, "beta": None,
                "source": "/work/dwong/jepa_a10_work/fixed_s9001/runs/"
                          "dev_s005_within__sigreg_jepa__s9001/final.pt",
                "sha256": "95cb5e3f4f158a719098a4b3dc573eebb8c71645c22290f6483223ccc04c484e"},
    "mae9000": {"arm": "mae_mse", "split": "within", "seed": 9000, "beta": None,
                "source": "/work/dwong/jepa_a10_work/dev_step2/runs/"
                          "dev_s005_within__mae_mse__s9000/final.pt",
                "sha256": "efd54c2586f8b5378dfe2febffe2ceb954c50185f6b07939c2b892c825270664"},
    "across9000": {"arm": "sigreg_jepa", "split": "across", "seed": 9000, "beta": None,
                   "source": "/ceph/dwong/jepa/returns/ext_pregrid_20261025/train/"
                             "across_s9000/runs/ext_dev_s005_across__sigreg_jepa__s9000/final.pt",
                   "sha256": "ac332d4b602d28fe2fdc2b4c1094058eed33538be4f0a9bc5c18503eac73bd5f",
                   "sibling": "/ceph/dwong/jepa/returns/ext_pregrid_20261025/train/"
                              "across_s9000/runs/ext_dev_s005_across__sigreg_jepa__s9000/"
                              "ext_summary.json",
                   "sibling_sha256": "708dc0092018fed09bdd29c7cf578e3586bd2277cafdacbbee005a88ad5c13cd"},
    "across9001": {"arm": "sigreg_jepa", "split": "across", "seed": 9001, "beta": None,
                   "source": "/ceph/dwong/jepa/returns/ext_pregrid_20261025/train/"
                             "across_s9001/runs/ext_dev_s005_across__sigreg_jepa__s9001/final.pt",
                   "sha256": "749e87376123ff629e8f2bf7f02f012f3ac965c181266b4710ce687c8055651a",
                   "sibling": "/ceph/dwong/jepa/returns/ext_pregrid_20261025/train/"
                              "across_s9001/runs/ext_dev_s005_across__sigreg_jepa__s9001/"
                              "ext_summary.json",
                   "sibling_sha256": "e72809b6dc39b7c6cf6a84f8222afbf627a3a56bc656f3e144494ea7f8985aaa"},
    "init9000": {"arm": "sigreg_jepa", "split": "within", "seed": 9000, "beta": None,
                 "source": None, "sha256": None},
}
ALL_ITEMS = tuple(ITEMS)


# ----------------------------------------------------------------------------- io
def _json_safe(value):
    return lr._safe(value)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=1, default=float))
    tmp.replace(path)


def write_plan(path: Path, payload: dict) -> None:
    write_json(path, payload)


# ------------------------------------------------------------------- setup/data
def load_setup(args):
    config = lr.load_run_config(getattr(args, "config", None) or CONFIG_PATH)
    gen_path = Path(getattr(args, "generator_config", None) or GENERATOR_PATH)
    gen = lb.load_generator_config(str(gen_path))
    arms_config = la.load_config()
    geometry_path = Path(getattr(args, "geometry", None) or GEOMETRY_DEFAULT)
    geometry = lb.load_wcte_geometry(str(geometry_path))
    device = lr.configure_runtime(getattr(args, "device", "auto") or "auto")
    return config, gen, arms_config, geometry, device


def config_hashes(args) -> dict:
    config_path = Path(getattr(args, "config", None) or CONFIG_PATH)
    gen_path = Path(getattr(args, "generator_config", None) or GENERATOR_PATH)
    got = {"config": {"path": str(config_path), "sha256": lr.sha256_file(config_path)},
           "generator_config": {"path": str(gen_path), "sha256": lr.sha256_file(gen_path)}}
    got["config"]["expected_sha256"] = CONFIG_SHA256
    got["generator_config"]["expected_sha256"] = GENERATOR_SHA256
    got["pass"] = (got["config"]["sha256"] == CONFIG_SHA256
                   and got["generator_config"]["sha256"] == GENERATOR_SHA256)
    return got


def load_probes(config, gen, geometry, data_root: Path) -> dict:
    """The two dev probe sets only (§3.4: never ``datasets_for``/``dev_train``)."""
    cell = lr.dev_cell(config)
    seed = int(config["data"]["dataset_seed"])
    common = dict(data_root=data_root, geometry=geometry, generator_config=gen, seed=seed,
                  code_sha=None, log=print)
    probes = {}
    for role, name in (("train", "dev_probe_train"), ("test", "dev_probe_test")):
        probes[role] = lr.ensure_dataset(cohort_spec=config["data"]["cohorts"][name],
                                         cohort_name=name, law="F1", setting=cell.setting,
                                         registered=True, **common)
    return probes


def preflight_data(config, gen, geometry, data_root: Path, *, regenerate: bool = True) -> dict:
    """§3.4 read-only probe-set preflight (regenerates only on a mismatch)."""
    data_root = Path(data_root)
    resolved = data_root.resolve()
    root_lucid = data_root / "lucid"
    out = {"registered": False, "extension": EXTENSION, "data_root": str(data_root),
           "resolved_root": str(resolved), "check": "§3.4 probe-set preflight",
           "probe_sets": {}, "regenerated": False}
    for name, expected in PROBE_MANIFESTS.items():
        directory = root_lucid / f"{name}__F1__0.05__s{int(config['data']['dataset_seed'])}"
        manifest = directory / "manifest.json"
        entry = {"directory": str(directory), "exists": directory.is_dir(),
                 "manifest": str(manifest), "exists_manifest": manifest.is_file()}
        if manifest.is_file():
            entry["manifest_sha256"] = lr.sha256_file(manifest)
            entry["expected_sha256"] = expected
            entry["pass"] = entry["manifest_sha256"] == expected
        else:
            entry["manifest_sha256"] = None
            entry["expected_sha256"] = expected
            entry["pass"] = False
        out["probe_sets"][name] = entry
    out["pass"] = all(entry["pass"] for entry in out["probe_sets"].values())
    out["dev_train_built"] = (root_lucid / f"dev_train__F1__0.05__s{int(config['data']['dataset_seed'])}"
                              "__n100000x1000" / "manifest.json").is_file()
    if not out["pass"]:
        if not regenerate:
            raise RuntimeError("§3.4 S9: probe-set preflight failed and regeneration is disabled")
        out["regenerated"] = _regenerate_probe_sets(config, gen, geometry, data_root, out)
    return out


def _regenerate_probe_sets(config, gen, geometry, data_root: Path, preflight: dict) -> True:
    """§3.4 fallback: archive the mismatching set inside ``<ext>`` and regenerate it.

    Deterministic regeneration (pinned generator config and ``dataset_seed``); the
    regenerated manifest must equal the provenance sha256 or this raises (S9).
    """
    ext = Path(preflight["directory"] if "directory" in preflight else ".").parent
    archive = Path(os.environ.get("JEPA_READOUT_EXT", str(ext))) / "data_archive"
    cell = lr.dev_cell(config)
    seed = int(config["data"]["dataset_seed"])
    for name, expected in PROBE_MANIFESTS.items():
        entry = preflight["probe_sets"][name]
        directory = Path(entry["directory"])
        if entry["pass"]:
            continue
        if directory.is_dir():
            archive.mkdir(parents=True, exist_ok=True)
            target = archive / f"{name}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
            shutil.move(str(directory), str(target))
        lr.ensure_dataset(cohort_spec=config["data"]["cohorts"][name], cohort_name=name,
                          law="F1", setting=cell.setting, registered=True,
                          data_root=data_root, geometry=geometry, generator_config=gen,
                          seed=seed, code_sha=None, verify_hashes=True, log=print)
        manifest = directory / "manifest.json"
        if not manifest.is_file() or lr.sha256_file(manifest) != expected:
            raise RuntimeError(f"§3.4 S9: regenerated {name} manifest does not match provenance "
                               f"({expected})")
        entry["manifest_sha256"] = lr.sha256_file(manifest)
        entry["pass"] = True
    preflight["pass"] = all(entry["pass"] for entry in preflight["probe_sets"].values())
    return True


# ------------------------------------------------------------- checkpoint manifest
def _fallback_sources(label: str, spec: dict) -> list[Path]:
    """§3.3 fallback: the /ceph copy in the same run family (if any)."""
    seed = int(spec["seed"])
    run_id = (f"ext_dev_s005_within__sigreg_jepa__s{seed:04d}" if spec["split"] == "within"
              else f"ext_dev_s005_across__sigreg_jepa__s{seed:04d}")
    prefix = {"old": "old", "fix": "fixed", "mae": "mae"}.get(label.split("9")[0])
    candidates = []
    if prefix:
        for family in ("old_s%d" % seed, "fixed_s%d" % seed, "mae_s%d" % seed,
                       "dev_step2", "s%d" % seed):
            candidates.append(Path("/ceph/dwong/jepa/returns/ext_pregrid_20261025/train") /
                              family / "runs" / run_id / "final.pt")
    return candidates


def verify_sibling(spec: dict) -> dict:
    """Verify an across item's sibling ``ext_summary.json`` against the D2 SHA256SUMS."""
    sibling, expected = spec.get("sibling"), spec.get("sibling_sha256")
    if not sibling:
        return {"required": False}
    path = Path(sibling)
    row = {"sibling": str(path), "expected_sha256": expected, "exists": path.is_file()}
    if not path.is_file():
        row["pass"] = False
        return row
    row["sha256"] = lr.sha256_file(path)
    row["pass"] = row["sha256"] == expected
    if D2_SHA256SUMS.is_file():
        listed = {}
        for line in D2_SHA256SUMS.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                listed[parts[1]] = parts[0]
        relative = str(path).split("/ceph/dwong/jepa/", 1)[-1]
        row["listed_in_d2_sha256sums"] = listed.get(relative)
        row["pass"] = bool(row["pass"] and row["listed_in_d2_sha256sums"] in (None, expected)
                           and (row["listed_in_d2_sha256sums"] is None
                                or row["listed_in_d2_sha256sums"] == expected))
    return row


def resolve_item(label: str, *, ckpt_dir: Path | None = None) -> dict:
    """Resolve, verify (and optionally copy) one §3.3 checkpoint item."""
    if label not in ITEMS:
        raise KeyError(f"unknown item {label!r}; expected one of {ALL_ITEMS}")
    spec = ITEMS[label]
    item = {"label": label, "arm": spec["arm"], "split": spec["split"],
            "seed": int(spec["seed"]), "beta": spec["beta"], "init": spec["source"] is None,
            "source": spec["source"], "expected_sha256": spec["sha256"],
            "sibling": verify_sibling(spec)}
    if item["init"]:
        item.update(source_sha256=None, ckpt=None, ckpt_sha256=None, missing=False)
        return item
    source = Path(spec["source"])
    if not source.is_file():
        for candidate in _fallback_sources(label, spec):
            if candidate.is_file():
                source = candidate
                break
    item["source"] = str(source)
    if not source.is_file():
        item.update(missing=True, source_sha256=None, ckpt=None, ckpt_sha256=None)
        return item
    source_sha = lr.sha256_file(source)
    item["source_sha256"] = source_sha
    if item["expected_sha256"] is not None and source_sha != item["expected_sha256"]:
        raise RuntimeError(f"{label}: checkpoint sha256 {source_sha} != manifest "
                           f"{item['expected_sha256']}; refusing to use it")
    if item["sibling"].get("required", True) and not item["sibling"].get("pass", True):
        raise RuntimeError(f"{label}: sibling ext_summary.json verification failed: "
                           f"{item['sibling']}")
    item["missing"] = False
    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        target = ckpt_dir / f"{label}.pt"
        if not (target.is_file() and lr.sha256_file(target) == source_sha):
            tmp = target.with_name(target.name + ".tmp")
            shutil.copyfile(source, tmp)
            os.replace(tmp, target)
        copy_sha = lr.sha256_file(target)
        if copy_sha != source_sha:
            raise RuntimeError(f"{label}: checkpoint copy hash {copy_sha} != source {source_sha}")
        item["ckpt"] = str(target)
        item["ckpt_sha256"] = copy_sha
    return item


def plan_table(labels, requested_forms) -> list:
    """The item x form table for the plan; half_pass is n/a on the across split."""
    rows = []
    for label in labels:
        split = ITEMS[label]["split"]
        applicable = er.WITHIN_FORMS if split == "within" else er.ACROSS_FORMS
        forms = [form for form in requested_forms if form in applicable]
        skipped = [form for form in requested_forms if form not in applicable]
        rows.append({"label": label, "split": split, "forms": forms,
                     "n_a": [{"form": form,
                              "reason": "half_pass needs the fixed within target set (§2: "
                                        "across is per-event random)"} for form in skipped]})
    return rows


def manifest_items(labels, *, ckpt_dir=None) -> dict:
    items, missing = {}, []
    for label in labels:
        item = resolve_item(label, ckpt_dir=ckpt_dir)
        items[label] = item
        if item.get("missing"):
            missing.append(label)
    priority = ("old9000", "old9001", "fix9000", "fix9001", "mae9000", "init9000")
    available = [label for label in priority if not items[label].get("missing") and label in items]
    if len(available) < 4:
        raise RuntimeError(f"§3.3 S9: fewer than four priority items available: {available}")
    return {"items": items, "missing": missing, "priority_available": available}


# --------------------------------------------------------------------- dry-run
def mode_dry_run(args) -> int:
    started = time.time()
    config, gen, arms_config, geometry, _ = load_setup(args)
    out = Path(args.out)
    ext = out.parent
    items_path = Path(getattr(args, "manifest", None) or (ext / "items.json"))
    ckpt_dir = Path(args.ckpt_dir or (ext / "ckpt"))
    labels = _split_labels(args.items) if args.items else list(ALL_ITEMS)
    forms = _split_labels(args.forms) if args.forms else list(er.WITHIN_FORMS)
    for form in forms:
        if form not in er.FORMS:
            raise SystemExit(f"unknown form {form!r}; expected {er.FORMS}")
    data = preflight_data(config, gen, geometry, Path(args.data_root or DATA_ROOT_DEFAULT))
    hashes = config_hashes(args)
    if not hashes["pass"]:
        raise SystemExit(f"config sha256 mismatch: {hashes}")
    manifest = manifest_items(labels, ckpt_dir=ckpt_dir)
    table = plan_table(labels, forms)
    plan = {"registered": False, "extension": EXTENSION, "mode": "dry-run",
            "runbook": "docs/jepa/RUNBOOK_2026-11-01_READOUT.md (revision 2)",
            "code_sha": lr.git_head(ROOT), "git_status": lr.git_status(ROOT),
            "config_sha256": hashes,
            "data": data,
            "checkpoints": {"dir": str(ckpt_dir),
                            "items": {label: {key: value for key, value in item.items()
                                              if key in ("source", "source_sha256", "ckpt",
                                                         "ckpt_sha256", "expected_sha256",
                                                         "init", "missing", "sibling")}
                                      for label, item in manifest["items"].items()},
                            "missing": manifest["missing"],
                            "priority_available": manifest["priority_available"]},
            "planned": table,
            "n_items": len(table),
            "n_forms": len(forms),
            "writes": {"plan": str(out), "items": str(items_path),
                       "checkpoints": str(ckpt_dir)},
            "wall_seconds": None}
    write_json(items_path, {"registered": False, "extension": EXTENSION,
                            "items": manifest["items"],
                            "missing": manifest["missing"],
                            "priority_available": manifest["priority_available"]})
    plan["wall_seconds"] = round(time.time() - started, 3)
    write_plan(out, plan)
    print(json.dumps({"plan": str(out), "items": [row["label"] for row in table],
                      "forms": forms, "data_pass": data["pass"],
                      "missing": manifest["missing"]}, default=str), flush=True)
    return 0


# ---------------------------------------------------------------------- matrix
def _save_form_cache(path: Path, train: er.FormStream, test: er.FormStream) -> None:
    """Cache the event/token rows ``nonlinear`` needs (tokens as float16 for size)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    np.savez(tmp,
             event_train=train.events.astype(np.float32),
             event_test=test.events.astype(np.float32),
             run_ids_train=np.asarray(train.run_ids, dtype=str),
             run_ids_test=np.asarray(test.run_ids, dtype=str),
             vertex_train=train.vertex.astype(np.float64),
             vertex_test=test.vertex.astype(np.float64),
             log_yield_train=train.log_yield.astype(np.float64),
             log_yield_test=test.log_yield.astype(np.float64),
             tokens_train=train.tokens.astype(np.float16),
             tokens_test=test.tokens.astype(np.float16),
             token_y_train=train.token_y.astype(np.float64),
             token_y_test=test.token_y.astype(np.float64),
             token_run_ids_train=np.asarray(train.token_run_ids, dtype=str),
             token_run_ids_test=np.asarray(test.token_run_ids, dtype=str))
    os.replace(tmp, path)


def _load_form_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _entry_table(metrics: dict) -> dict:
    """Keep the matrix.json cells small and explicit (value or n/a + reason)."""
    token = metrics.get("token_log_charge_r2") or {}
    gain = metrics.get("module_gain_r2") or {}
    token_cell = ({"r2": token.get("r2"), "alpha": token.get("alpha"),
                   "n_train": token.get("n_train"), "n_test": token.get("n_test")}
                  if token.get("status") == "evaluated"
                  else {"value": None, "reason": token.get("reason", "n/a")})
    gain_cell = ({"value": gain.get("value"), "source": gain.get("source")}
                 if gain.get("status") == "evaluated"
                 else {"value": None, "reason": gain.get("reason", "n/a")})
    return {"vertex_resolution_m": metrics.get("vertex_resolution_m"),
            "vertex": metrics.get("vertex"),
            "token_log_charge_r2": token_cell,
            "effective_rank": metrics.get("effective_rank"),
            "participation_ratio": metrics.get("participation_ratio"),
            "event_sigreg": metrics.get("event_sigreg"),
            "module_gain_r2": gain_cell,
            "layout": metrics.get("layout"),
            "n_events": metrics.get("n_events"),
            "n_token_rows": metrics.get("n_token_rows")}


def mode_matrix(args) -> int:
    started = time.time()
    config, gen, arms_config, geometry, device = load_setup(args)
    out = Path(args.out)
    ext = out.parent
    manifest_path = Path(args.manifest or (ext / "items.json"))
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest {manifest_path}; run dry-run first")
    manifest = json.loads(manifest_path.read_text())
    labels = _split_labels(args.items) if args.items else list(er.GROUP_A)
    for label in labels:
        if label not in manifest["items"]:
            raise SystemExit(f"{label} is not in {manifest_path}")
    requested = _split_labels(args.forms) if args.forms else list(er.WITHIN_FORMS)
    for form in requested:
        if form not in er.FORMS:
            raise SystemExit(f"unknown form {form!r}; expected {er.FORMS}")
    ckpt_dir = Path(args.ckpt_dir or (ext / "ckpt"))
    cache_dir = Path(args.cache_dir or (ext / "cache"))
    data_root = Path(args.data_root or DATA_ROOT_DEFAULT)
    probes = load_probes(config, gen, geometry, data_root)
    group = args.group or out.name.replace("matrix_", "") or "matrix"
    mask_seed = int(config["views"]["mask_seed"])
    chunk_events = int(args.chunk_events)
    report = {"registered": False, "extension": EXTENSION, "mode": "matrix", "group": str(group),
              "runbook": "docs/jepa/RUNBOOK_2026-11-01_READOUT.md (revision 2)",
              "code_sha": lr.git_head(ROOT), "config_sha256": CONFIG_SHA256,
              "device": str(device), "host": os.uname().nodename,
              "chunk_events": chunk_events, "domain": er.PRIMARY_DOMAIN,
              "probe_sets": {role: probes[role].key for role in ("train", "test")},
              "probe_manifest_sha256": {role: probes[role].manifest_sha256()
                                        for role in ("train", "test")},
              "items": {}, "complete": False}
    for label in labels:
        item_started = time.time()
        item = dict(manifest["items"][label])
        if item.get("missing"):
            report["items"][label] = {"missing": True}
            continue
        if item.get("ckpt"):
            source = Path(item["ckpt"])
            if not source.is_file() or lr.sha256_file(source) != item["ckpt_sha256"]:
                raise RuntimeError(f"{label}: verified checkpoint copy is missing or changed")
            item["source"] = str(source)
        split = item["split"]
        applicable = er.WITHIN_FORMS if split == "within" else er.ACROSS_FORMS
        forms = [form for form in requested if form in applicable]
        arm = er.item_arm(item, config, arms_config, geometry, device=device)
        entry = {"arm": item["arm"], "split": split, "seed": int(item["seed"]),
                 "source_sha256": item.get("source_sha256"),
                 "ckpt_sha256": item.get("ckpt_sha256"), "forms": {}}
        full_forms = [form for form in forms if form != "half_pass"]
        half_forms = [form for form in forms if form == "half_pass"]
        if full_forms:
            streamed = er.stream_model_forms(
                arm.jepa, probes, full_forms, geometry, split=split, mask_seed=mask_seed,
                device=device, chunk_events=chunk_events, max_events=args.probe_max_events,
                domain=er.PRIMARY_DOMAIN)
            for form in full_forms:
                metrics = er.form_metrics(streamed["train"][form], streamed["test"][form],
                                          module_gain=streamed["module_gain"])
                entry["forms"][form] = _entry_table(metrics)
                if form in ("full_input", "half_pass"):
                    _save_form_cache(cache_dir / f"{label}__{form}.npz",
                                     streamed["train"][form], streamed["test"][form])
                print(f"[matrix {group}] {label} {form}: vertex="
                      f"{metrics['vertex_resolution_m']} lq="
                      f"{(metrics.get('token_log_charge_r2') or {}).get('r2')}",
                      flush=True)
        if half_forms:
            half = er.HalfPassEncoder(arm.jepa, er.within_target_mask(geometry)).to(device)
            streamed = er.stream_model_forms(
                half, probes, ["half_pass"], geometry, split=split, mask_seed=mask_seed,
                device=device, chunk_events=chunk_events, max_events=args.probe_max_events,
                domain=er.PRIMARY_DOMAIN)
            metrics = er.form_metrics(streamed["train"]["half_pass"],
                                      streamed["test"]["half_pass"],
                                      module_gain=streamed["module_gain"])
            entry["forms"]["half_pass"] = _entry_table(metrics)
            _save_form_cache(cache_dir / f"{label}__half_pass.npz",
                             streamed["train"]["half_pass"], streamed["test"]["half_pass"])
            print(f"[matrix {group}] {label} half_pass: vertex="
                  f"{metrics['vertex_resolution_m']} lq="
                  f"{(metrics.get('token_log_charge_r2') or {}).get('r2')}", flush=True)
        for form in requested:
            if form not in entry["forms"]:
                entry["forms"][form] = {
                    "vertex_resolution_m": None, "vertex": {"status": "n/a",
                                                            "reason": "half_pass needs the "
                                                                      "fixed within target set"},
                    "token_log_charge_r2": {"value": None, "reason": "n/a on the across split"},
                    "effective_rank": None, "participation_ratio": None, "event_sigreg": None,
                    "module_gain_r2": {"value": None, "reason": "n/a on the across split"},
                    "layout": None, "n_events": {"train": 0, "test": 0},
                    "n_token_rows": {"train": 0, "test": 0}}
        del arm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        report["items"][label] = entry
        report["items"][label]["wall_seconds"] = round(time.time() - item_started, 3)
        write_json(out / "matrix.json", report)   # keep finished items on a crash
    report["complete"] = True
    report["wall_seconds"] = round(time.time() - started, 3)
    write_json(out / "matrix.json", report)
    expected = {key: value for key, value in er.REPRODUCTION.items() if key in labels}
    reproduction = er.reproduction_block(report["items"], expected=expected,
                                         tolerance=float(args.reproduction_tolerance))
    reproduction["group"] = str(group)
    write_json(out / f"reproduction_{group}.json", reproduction)
    print(json.dumps({"group": group, "reproduction_pass": reproduction["reproduction_pass"],
                      "items": sorted(report["items"])}, default=float), flush=True)
    if not reproduction["reproduction_pass"]:
        print(f"[matrix {group}] §4.2 reproduction FAILED: "
              f"{json.dumps(reproduction['items'], default=float)}", file=sys.stderr, flush=True)
        return 2
    return 0


# ------------------------------------------------------------------- nonlinear
def _cache_split(data: dict, split: str) -> dict:
    """One cached split's event/token rows, with the tokens upcast to float32."""
    return {"event": np.asarray(data[f"event_{split}"], np.float32),
            "vertex": np.asarray(data[f"vertex_{split}"], np.float64),
            "run_ids": np.asarray(data[f"run_ids_{split}"], dtype=object),
            "log_yield": np.asarray(data[f"log_yield_{split}"], np.float64),
            "tokens": np.asarray(data[f"tokens_{split}"], np.float64),
            "token_y": np.asarray(data[f"token_y_{split}"], np.float64),
            "token_run_ids": np.asarray(data[f"token_run_ids_{split}"], dtype=object)}


def _vertex_from_cache(train: dict, test: dict, form: str) -> tuple[dict, dict]:
    dataset = lambda x, y, runs: la.ProbeDataset(
        kind="vertex", form=form, x=np.asarray(x, np.float64), y=np.asarray(y, np.float64),
        run_ids=np.asarray(runs, dtype=object), target_names=("x", "y", "z"), meta={})
    train_ds = dataset(train["event"], train["vertex"], train["run_ids"])
    test_ds = dataset(test["event"], test["vertex"], test["run_ids"])
    fitted = la.fit_probe(train_ds)
    predictions = la.probe_predictions(fitted, test_ds)
    scored = lr._vertex_error(predictions, test_ds.y)
    scored["alphas"] = [row["alpha"] for row in fitted["fits"] if row["status"] == "fit"]
    return scored, {"n_train": train_ds.n_rows, "n_test": test_ds.n_rows}


def mode_nonlinear(args) -> int:
    started = time.time()
    out = Path(args.out)
    matrix_dirs = [Path(path) for path in args.matrix]
    matrix_items, matrix_source = {}, {}
    for directory in matrix_dirs:
        payload = json.loads((directory / "matrix.json").read_text())
        for label, entry in payload["items"].items():
            matrix_items[label] = entry
            matrix_source[label] = str(directory)
    labels = _split_labels(args.items) if args.items else list(er.GROUP_A)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        matrix_dirs[0].parent / "cache")
    device = torch.device("cpu")
    code_sha = lr.git_head(ROOT)
    report = {"registered": False, "extension": EXTENSION, "mode": "nonlinear",
              "runbook": "docs/jepa/RUNBOOK_2026-11-01_READOUT.md (revision 2)",
              "code_sha": code_sha, "seed": int(args.seed), "device": "cpu",
              "matrix_sources": {label: matrix_source.get(label) for label in labels},
              "items": {}, "cka": {}, "complete": False}
    reps = {}
    for label in labels:
        if label not in matrix_items:
            raise SystemExit(f"{label} is not in any --matrix payload")
        cached = {}
        for form in ("full_input", "half_pass"):
            path = cache_dir / f"{label}__{form}.npz"
            if path.is_file():
                cached[form] = _load_form_cache(path)
        if not cached:
            raise SystemExit(f"no cached representations for {label} under {cache_dir}")
        entry = {"forms": {}}
        for form, data in cached.items():
            train = _cache_split(data, "train")
            test = _cache_split(data, "test")
            vertex_ridge, counts = _vertex_from_cache(train, test, form)
            vertex_mlp = er.mlp_probe(train["event"], train["vertex"],
                                      test["event"], test["vertex"], seed=int(args.seed),
                                      metric="resolution")
            lq_ridge = er.ridge_r2(train["tokens"], train["token_y"],
                                   test["tokens"], test["token_y"])
            lq_mlp = er.mlp_probe(train["tokens"], train["token_y"],
                                  test["tokens"], test["token_y"], seed=int(args.seed),
                                  metric="r2")
            entry["forms"][form] = {
                "vertex_ridge": {"resolution": vertex_ridge.get("resolution"),
                                 "median": vertex_ridge.get("median"),
                                 "n": vertex_ridge.get("n"),
                                 "alphas": vertex_ridge.get("alphas"),
                                 "n_train": counts["n_train"], "n_test": counts["n_test"]},
                "vertex_mlp": vertex_mlp,
                "token_log_charge_ridge": lq_ridge,
                "token_log_charge_mlp": lq_mlp,
                "n_token_rows": {"train": int(len(train["token_y"])),
                                 "test": int(len(test["token_y"]))},
                "matrix_check": {
                    "vertex_ridge": ((matrix_items[label].get("forms", {})
                                      .get(form, {}) or {}).get("vertex_resolution_m")),
                    "token_log_charge_ridge": (((matrix_items[label].get("forms", {})
                                               .get(form, {}) or {})
                                              .get("token_log_charge_r2", {}) or {}).get("r2"))}}
            print(f"[nonlinear] {label} {form}: ridge_vertex="
                  f"{vertex_ridge.get('resolution')} mlp_vertex="
                  f"{vertex_mlp.get('resolution')} ridge_lq={lq_ridge.get('r2')} "
                  f"mlp_lq={lq_mlp.get('r2_mean')}", flush=True)
        entry["split"] = (matrix_items[label] or {}).get("split")
        report["items"][label] = entry
        reps[label] = cached["full_input"]["event_test"] if "full_input" in cached else None
        write_json(out / "nonlinear.json", report)
    # Q2-d CKA rows: event-level full_input means on the same probe-test rows.
    if "init9000" in reps and reps["init9000"] is not None:
        init = reps["init9000"]
        init_cache = _load_form_cache(cache_dir / "init9000__full_input.npz")
        raw_physics = np.column_stack([init_cache["vertex_test"],
                                       init_cache["log_yield_test"]])
        report["cka"]["raw_physics_summary"] = "[vertex x, y, z, log yield] per event"
        report["cka"]["init9000_vs_raw_physics"] = er.cka(init, raw_physics)
        for label in labels:
            if reps.get(label) is None or label == "init9000":
                continue
            report["cka"][f"{label}_vs_init9000"] = er.cka(reps[label], init)
            report["cka"][f"{label}_vs_raw_physics"] = er.cka(reps[label], raw_physics)
    report["complete"] = True
    report["wall_seconds"] = round(time.time() - started, 3)
    write_json(out / "nonlinear.json", report)
    print(json.dumps({"complete": True, "items": sorted(report["items"])}, default=float),
          flush=True)
    return 0


# ---------------------------------------------------------------------- report
def _fmt(value, digits=6):
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _readings(matrix_items: dict, nonlinear: dict) -> dict:
    """§6 pre-declared readings from the merged matrix and nonlinear payloads."""
    def form_value(label, form, metric):
        entry = (matrix_items.get(label, {}).get("forms", {}) or {}).get(form, {}) or {}
        return entry.get(metric)

    def vertex(label, form):
        return form_value(label, form, "vertex_resolution_m")

    def lq(label, form):
        cell = (matrix_items.get(label, {}).get("forms", {}) or {}).get(form, {}).get(
            "token_log_charge_r2", {}) or {}
        return cell.get("r2") if isinstance(cell, dict) else None

    def spread(values):
        finite = [value for value in values if value is not None]
        return (max(finite) - min(finite)) if len(finite) == len(values) else None

    init_full = vertex("init9000", "full_input")
    init_half = vertex("init9000", "half_pass")
    readings = {}
    # Q1-a: full-input worse than the random-init floor while half-pass is not.
    rows, conds = {}, {}
    for label in ("fix9000", "fix9001"):
        full_v, half_v = vertex(label, "full_input"), vertex(label, "half_pass")
        full_ratio = (full_v / init_full) if (full_v is not None and init_full) else None
        half_ratio = (half_v / init_half) if (half_v is not None and init_half) else None
        holds = bool(full_ratio is not None and half_ratio is not None
                     and full_ratio > 1.1 and half_ratio <= 1.1)
        conds[label] = holds
        rows[label] = {"full_input_vertex_m": full_v, "init_full_input_vertex_m": init_full,
                       "full_ratio": full_ratio,
                       "half_pass_vertex_m": half_v, "init_half_pass_vertex_m": init_half,
                       "half_ratio": half_ratio, "holds": holds}
    if all(conds.values()):
        verdict = "supported"
    elif not any(conds.values()):
        verdict = "not supported"
    else:
        verdict = "unclear (the two seeds disagree)"
    readings["Q1-a"] = {"question": "trained-vs-random verdict is readout-dependent",
                        "supported_if": "full-input vertex > 1.1 x init floor AND half-pass "
                                        "vertex <= 1.1 x half-pass init floor, both seeds",
                        "verdict": verdict, "items": rows}
    # Q1-b: the half-pass fix-vs-old difference is not seed noise.
    rows = {}
    for seed in (9000, 9001):
        fix_v, old_v = vertex(f"fix{seed}", "half_pass"), vertex(f"old{seed}", "half_pass")
        diff = (fix_v - old_v) if (fix_v is not None and old_v is not None) else None
        spread_fix = spread([vertex(f"fix{seed}", "half_pass") for seed in (9000, 9001)])
        spread_old = spread([vertex(f"old{seed}", "half_pass") for seed in (9000, 9001)])
        known = [value for value in (spread_fix, spread_old) if value is not None]
        pair_spread = max(known) if known else None
        rows[str(seed)] = {"half_pass_vertex_fix_m": fix_v, "half_pass_vertex_old_m": old_v,
                           "difference_m": diff, "seed_spread_m": pair_spread,
                           "spread_fix_m": spread_fix, "spread_old_m": spread_old,
                           "difference_gt_spread": bool(diff is not None
                                                        and pair_spread is not None
                                                        and diff > pair_spread)}
    flags = [row["difference_gt_spread"] for row in rows.values()]
    diffs = [row["difference_m"] for row in rows.values()]
    if all(flags):
        verdict = "supported"
    elif all(value is not None and value > 0 for value in diffs):
        verdict = "unclear (one pair fails by less than the spread)"
    else:
        verdict = "not supported"
    readings["Q1-b"] = {"question": "half-pass label-version difference is not seed noise",
                        "supported_if": "half-pass vertex(fix) - half-pass vertex(old) > the "
                                        "pair's seed spread, both pairs",
                        "verdict": verdict, "seeds": rows,
                        "note": "the pair's seed spread is the larger of the fix/old half-pass "
                                "seed spreads (§6 definition of 'worse by more than the spread')"}
    # Q1-c: propagated carries the same information as half_pass.
    pairs = (("old", "fix"), ("old", "mae"), ("fix", "mae"))
    pair_rows, pair_ok = {}, True
    for left, right in pairs:
        row = {"pair": [left, right], "seeds": {}}
        for form in ("half_pass", "propagated"):
            diffs = []
            for seed in (9000, 9001):
                lv = vertex(f"{left}{seed}", form)
                rv = vertex(f"{right}{seed}", form)
                if lv is not None and rv is not None:
                    diffs.append(lv - rv)
            mean_diff = float(np.mean(diffs)) if diffs else None
            seed_spread = ((max(diffs) - min(diffs)) if len(diffs) >= 2 else None)
            spreads = [spread([vertex(f"{item}{seed}", form) for seed in (9000, 9001)])
                       for item in (left, right)]
            known = [value for value in spreads if value is not None]
            spread_form = max(known) if known else None
            row["seeds"][form] = {"seed_differences_m": diffs, "mean_difference_m": mean_diff,
                                  "difference_seed_spread_m": seed_spread,
                                  "item_seed_spread_m": spread_form}
        for form in ("half_pass", "propagated"):
            cell = row["seeds"][form]
            cell["exceeds_2x_spread"] = bool(cell["mean_difference_m"] is not None
                                             and cell["item_seed_spread_m"] is not None
                                             and abs(cell["mean_difference_m"])
                                             > 2 * cell["item_seed_spread_m"])
            cell["same_sign_both_seeds"] = (len(cell["seed_differences_m"]) == 2
                                            and np.sign(cell["seed_differences_m"][0])
                                            == np.sign(cell["seed_differences_m"][1]))
        half_cell, prop_cell = row["seeds"]["half_pass"], row["seeds"]["propagated"]
        same_sign = bool(half_cell["mean_difference_m"] is not None
                         and prop_cell["mean_difference_m"] is not None
                         and np.sign(half_cell["mean_difference_m"])
                         == np.sign(prop_cell["mean_difference_m"]))
        row["same_sign"] = same_sign
        row["holds"] = bool(same_sign and half_cell["exceeds_2x_spread"]
                            and prop_cell["exceeds_2x_spread"])
        pair_ok = pair_ok and row["holds"]
        pair_rows[f"{left}|{right}"] = row
    readings["Q1-c"] = {"question": "propagated carries the same information as half_pass",
                        "supported_if": "same sign of the vertex difference in both forms and "
                                        "the mean difference exceeds 2 x the seed spread, every "
                                        "pair among {old, fix, mae}",
                        "verdict": "supported" if pair_ok else "not supported",
                        "pairs": pair_rows,
                        "note": "seed spread = the larger of the two items' seed spreads in "
                                "that form (§6 definition); the mean is over the two seeds"}
    # Q1-e / Q1-f (descriptive).
    readings["Q1-e"] = {"question": "deterministic-layout signal (H1)",
                        "kind": "descriptive",
                        "values": {label: (form_value(label, "full_input", "layout") or {}).get(
                            "f_layout") for label in matrix_items}}
    readings["Q1-f"] = {"question": "event-dependence signal (H2)", "kind": "descriptive",
                        "values": {label: (form_value(label, "full_input", "layout") or {}).get(
                            "f_event") for label in matrix_items}}
    # Q2 rows from nonlinear, both seeds.
    def mlp(label, form, key):
        return (((nonlinear.get("items", {}).get(label, {}).get("forms", {}) or {})
                 .get(form, {}) or {}).get(key) or {})

    row_a, row_b, row_c, row_d = {}, {}, {}, {}
    for seed in (9000, 9001):
        label = f"fix{seed}"
        mlp_lq = mlp(label, "full_input", "token_log_charge_mlp")
        ridge_lq = mlp(label, "full_input", "token_log_charge_ridge")
        init_ridge = (nonlinear.get("items", {}).get("init9000", {}).get("forms", {})
                      .get("full_input", {}).get("token_log_charge_ridge", {}) or {})
        fix_mlp_vertex = mlp(label, "full_input", "vertex_mlp").get("resolution")
        init_mlp_vertex = (nonlinear.get("items", {}).get("init9000", {}).get("forms", {})
                           .get("full_input", {}).get("vertex_mlp", {}) or {}).get("resolution")
        mlp_r2 = mlp_lq.get("r2_mean")
        ridge_r2 = ridge_lq.get("r2")
        row_a[str(seed)] = {"mlp_token_r2": mlp_r2, "ridge_token_r2": ridge_r2,
                            "init_ridge_token_r2": init_ridge.get("r2"),
                            "holds": bool(mlp_r2 is not None and ridge_r2 is not None
                                          and init_ridge.get("r2") is not None
                                          and mlp_r2 <= 0.1 and ridge_r2 <= 0.05
                                          and init_ridge.get("r2") >= 0.9)}
        if mlp_r2 is not None and ridge_r2 is not None:
            if mlp_r2 >= 0.5 and ridge_r2 <= 0.1:
                b_verdict = "supported"
            elif 0.1 < mlp_r2 < 0.5:
                b_verdict = "unclear (MLP R2 between 0.1 and 0.5)"
            else:
                b_verdict = "not supported"
        else:
            b_verdict = "not evaluable"
        row_b[str(seed)] = {"mlp_token_r2": mlp_r2, "ridge_token_r2": ridge_r2,
                            "verdict": b_verdict}
        ratio = (fix_mlp_vertex / init_mlp_vertex
                 if fix_mlp_vertex is not None and init_mlp_vertex else None)
        row_c[str(seed)] = {"fix_mlp_vertex_m": fix_mlp_vertex,
                            "init_mlp_vertex_m": init_mlp_vertex, "ratio": ratio,
                            "holds": bool(ratio is not None and ratio > 1.1)}
        cka = nonlinear.get("cka", {})
        vs_init = cka.get(f"{label}_vs_init9000")
        vs_raw = cka.get(f"{label}_vs_raw_physics")
        diff = (vs_raw - vs_init) if (vs_init is not None and vs_raw is not None) else None
        if diff is None:
            d_verdict = "not evaluable"
        elif diff > 0.1:
            d_verdict = "supported"
        elif abs(diff) <= 0.1:
            d_verdict = "unclear (within +/- 0.1)"
        else:
            d_verdict = "not supported"
        row_d[str(seed)] = {"cka_vs_raw_physics": vs_raw, "cka_vs_init9000": vs_init,
                            "difference": diff, "verdict": d_verdict}
    def overall(rows, key="holds"):
        flags = [row.get(key) for row in rows.values()]
        return "supported" if flags and all(flags) else "not supported"
    readings["Q2-a"] = {"question": "charge is destroyed, not just unreadable",
                        "supported_if": "MLP token ln q R2(fix, full_input) <= 0.1 AND ridge "
                                        "<= 0.05 AND init ridge >= 0.9",
                        "verdict": overall(row_a), "seeds": row_a}
    by_verdict = {row["verdict"] for row in row_b.values()}
    readings["Q2-b"] = {"question": "charge is present but nonlinearly encoded",
                        "supported_if": "MLP token ln q R2(fix, full_input) >= 0.5 AND ridge "
                                        "<= 0.1 (unclear 0.1-0.5)",
                        "verdict": ("supported" if by_verdict == {"supported"}
                                    else ("unclear" if any("unclear" in value
                                                           for value in by_verdict)
                                          else "not supported")),
                        "seeds": row_b}
    readings["Q2-c"] = {"question": "vertex information is destroyed",
                        "supported_if": "MLP vertex resolution(fix) > 1.1 x init MLP vertex",
                        "verdict": overall(row_c), "seeds": row_c}
    d_verdicts = {row["verdict"] for row in row_d.values()}
    readings["Q2-d"] = {"question": "trained representation closer to physics than to a random "
                                    "projection",
                        "supported_if": "CKA(trained, raw physics) - CKA(trained, init) > 0.1 "
                                        "(unclear within +/-0.1)",
                        "verdict": ("supported" if d_verdicts == {"supported"}
                                    else ("unclear" if any("unclear" in value
                                                           for value in d_verdicts)
                                          else "not supported")),
                        "seeds": row_d}
    return readings


def mode_report(args) -> int:
    started = time.time()
    out = Path(args.out)
    matrix_items, matrix_groups, reproductions = {}, {}, []
    for directory in [Path(path) for path in args.matrix]:
        payload = json.loads((directory / "matrix.json").read_text())
        if not payload.get("complete"):
            raise SystemExit(f"{directory}/matrix.json is not complete")
        matrix_groups[str(directory)] = {"group": payload.get("group"),
                                         "code_sha": payload.get("code_sha"),
                                         "probe_manifest_sha256": payload.get(
                                             "probe_manifest_sha256")}
        for label, entry in payload["items"].items():
            if label in matrix_items:
                raise SystemExit(f"duplicate item {label} across matrix jobs")
            matrix_items[label] = entry
        for reproduction in sorted(directory.glob("reproduction_*.json")):
            reproductions.append(json.loads(reproduction.read_text()))
    nonlinear = json.loads(Path(args.nonlinear).read_text())
    if not nonlinear.get("complete"):
        raise SystemExit(f"{args.nonlinear} is not complete")
    merged = {"registered": False, "extension": EXTENSION,
              "runbook": "docs/jepa/RUNBOOK_2026-11-01_READOUT.md (revision 2)",
              "code_sha": lr.git_head(ROOT), "groups": matrix_groups,
              "probe_sets": {"dev_probe_train": er.PROBE_MANIFESTS["dev_probe_train"],
                             "dev_probe_test": er.PROBE_MANIFESTS["dev_probe_test"]},
              "items": matrix_items}
    reproduction = {"registered": False, "extension": EXTENSION, "check": "§4.2",
                    "tolerance_m": er.REPRODUCTION_TOLERANCE_M,
                    "groups": reproductions,
                    "reproduction_pass": all(row.get("reproduction_pass")
                                             for row in reproductions)
                    and bool(reproductions)}
    readings = _readings(matrix_items, nonlinear)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "matrix.json", merged)
    write_json(out / "nonlinear.json", nonlinear)
    write_json(out / "reproduction.json", reproduction)
    write_json(out / "readings.json", {"registered": False, "extension": EXTENSION,
                                       "readings": readings})
    sha_rows = {}
    for label, entry in matrix_items.items():
        sha_rows[label] = {"source_sha256": entry.get("source_sha256"),
                           "ckpt_sha256": entry.get("ckpt_sha256"),
                           "arm": entry.get("arm"), "split": entry.get("split"),
                           "seed": entry.get("seed")}
    readme = _readme(merged, nonlinear, readings, reproduction, sha_rows)
    (out / "README.md").write_text(readme)
    sizes = {path.name: path.stat().st_size for path in sorted(out.glob("*.json"))}
    too_big = {name: size for name, size in sizes.items() if size > 1_000_000}
    print(json.dumps({"out": str(out), "sizes": sizes, "too_big": too_big,
                      "reproduction_pass": reproduction["reproduction_pass"],
                      "readings": {key: value["verdict"] if "verdict" in value else "descriptive"
                                   for key, value in readings.items()},
                      "wall_seconds": round(time.time() - started, 3)}, default=float),
          flush=True)
    return 0 if reproduction["reproduction_pass"] and not too_big else 2


def _readme(merged: dict, nonlinear: dict, readings: dict, reproduction: dict,
            sha_rows: dict) -> str:
    lines = [
        "# Q1/Q2 readout matrix and nonlinear probes — 1 Nov 2026",
        "",
        "`registered: false`. Extension `readout_diagnostic_2026-11-01`.",
        "Runbook: `docs/jepa/RUNBOOK_2026-11-01_READOUT.md` (revision 2, "
        "sha256 `0e629141fc90d24cfa5bf3b0d1ac2ac271e7c94738eca68c678393032ee58c83`).",
        "Evaluation only: no encoder was trained, no grid run; photon domain only.",
        "",
        f"- final code SHA: `{merged.get('code_sha')}`",
        f"- §4.2 reproduction: {'PASS' if reproduction['reproduction_pass'] else 'FAIL'} "
        f"(tolerance {reproduction['tolerance_m']} m)",
        "",
        "## Checkpoint hashes used (§3.3)",
        "",
        "| item | arm | split | seed | source sha256 | copy sha256 |",
        "|---|---|---|---:|---|---|",
    ]
    for label, row in sha_rows.items():
        lines.append(f"| {label} | {row['arm']} | {row['split']} | {row['seed']} | "
                     f"`{_fmt(row['source_sha256'])}` | `{_fmt(row['ckpt_sha256'])}` |")
    lines += ["", "## Readout matrix (photon domain, Part-C-comparable protocol)", "",
              "| item | form | vertex m | token ln q R2 | eff. rank | part. ratio | "
              "event SIGReg | module gain R2 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for label, entry in merged["items"].items():
        for form, cell in (entry.get("forms") or {}).items():
            token = cell.get("token_log_charge_r2") or {}
            gain = cell.get("module_gain_r2") or {}
            lines.append(
                f"| {label} | {form} | {_fmt(cell.get('vertex_resolution_m'))} | "
                f"{_fmt(token.get('r2'))} | {_fmt(cell.get('effective_rank'))} | "
                f"{_fmt(cell.get('participation_ratio'))} | {_fmt(cell.get('event_sigreg'))} | "
                f"{_fmt(gain.get('value'))} |")
    lines += ["", "`n/a` cells and their reasons are in `matrix.json`; module gain is only "
              "defined for `full_input`/`half_pass` (`module_gain::run_mean`) and "
              "`propagated` (`module_gain::propagated`), §2.", "",
              "## §4.2 reproduction", "",
              "| item | recorded Part C m | recomputed m | |diff| m | pass |",
              "|---|---:|---:|---:|---|"]
    for group in reproduction["groups"]:
        for label, row in group["items"].items():
            lines.append(f"| {label} | {_fmt(row['expected_m'])} | {_fmt(row['got_m'])} | "
                         f"{_fmt(row['abs_diff_m'])} | {row['pass']} |")
    lines += ["", "## §6 pre-declared readings", ""]
    for key, value in readings.items():
        if "verdict" in value:
            lines.append(f"- **{key}** — {value['verdict']} ({value['question']}).")
        else:
            lines.append(f"- **{key}** — descriptive ({value['question']}): "
                         f"{json.dumps(value.get('values'), default=float)}.")
    lines += ["", "## Q2 probe table", "",
              "| item | form | ridge vertex m | MLP vertex m | ridge token R2 | MLP token R2 |",
              "|---|---|---:|---:|---:|---:|"]
    for label, entry in nonlinear.get("items", {}).items():
        for form, cell in (entry.get("forms") or {}).items():
            lines.append(
                f"| {label} | {form} | "
                f"{_fmt((cell.get('vertex_ridge') or {}).get('resolution'))} | "
                f"{_fmt((cell.get('vertex_mlp') or {}).get('resolution'))} | "
                f"{_fmt((cell.get('token_log_charge_ridge') or {}).get('r2'))} | "
                f"{_fmt((cell.get('token_log_charge_mlp') or {}).get('r2_mean'))} |")
    lines += ["", "## CKA (full_input event-level means, same probe-test rows)", ""]
    for key, value in (nonlinear.get("cka") or {}).items():
        lines.append(f"- `{key}` = {_fmt(value)}")
    lines += ["", "## Outputs", "",
              "- `matrix.json` — merged matrix_a/matrix_b, every form cell with value or "
              "explicit n/a + reason.",
              "- `nonlinear.json` — Q2-a/b/c ridge-vs-MLP on identical cached rows and the "
              "Q2-d CKA rows.",
              "- `reproduction.json` — the two §4.2 reproduction blocks.",
              "- `readings.json` — the §6 readings with their numbers (machine-readable).",
              "- `README.md` — this file.", ""]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ parser
def _split_labels(value: str) -> list:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _common(p):
    p.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--generator-config", default=str(GENERATOR_PATH))
    p.add_argument("--geometry", default=GEOMETRY_DEFAULT)


def parser():
    p = argparse.ArgumentParser(prog="jepa_ext_readout")
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dry-run")
    d.add_argument("--out", required=True)
    d.add_argument("--manifest", default=None)
    d.add_argument("--items", default=None)
    d.add_argument("--forms", default=None)
    d.add_argument("--ckpt-dir", default=None)
    _common(d)
    d.add_argument("--device", default="cpu")
    m = sub.add_parser("matrix")
    m.add_argument("--items", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--forms", required=True)
    m.add_argument("--manifest", default=None)
    m.add_argument("--ckpt-dir", default=None)
    m.add_argument("--cache-dir", default=None)
    m.add_argument("--group", default=None)
    m.add_argument("--chunk-events", type=int, default=32)
    m.add_argument("--probe-max-events", type=int, default=None,
                   help="engineering debug cap (never used for the reported run)")
    m.add_argument("--reproduction-tolerance", type=float, default=er.REPRODUCTION_TOLERANCE_M)
    m.add_argument("--device", default="auto")
    _common(m)
    n = sub.add_parser("nonlinear")
    n.add_argument("--matrix", action="append", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--items", default=None)
    n.add_argument("--cache-dir", default=None)
    n.add_argument("--seed", type=int, default=er.SAMPLE_SEED)
    _common(n)
    n.add_argument("--device", default="cpu")
    r = sub.add_parser("report")
    r.add_argument("--matrix", action="append", required=True)
    r.add_argument("--nonlinear", required=True)
    r.add_argument("--out", required=True)
    _common(r)
    r.add_argument("--device", default="cpu")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode == "dry-run":
        return mode_dry_run(args)
    if args.mode == "matrix":
        return mode_matrix(args)
    if args.mode == "nonlinear":
        return mode_nonlinear(args)
    return mode_report(args)


if __name__ == "__main__":
    sys.exit(main())
